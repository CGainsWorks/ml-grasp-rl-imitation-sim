"""Serve grasp actions from an exported TorchScript policy over ROS 2 topics.

Subscribes
    ``grasp_policy/observation``  std_msgs/Float32MultiArray, the 32 values of
                                  the observation layout in the policy's
                                  ``.meta.json`` (``docs/ros2-policy-server.md``)
Publishes
    ``grasp_policy/action``       std_msgs/Float32MultiArray, 4 values in
                                  [-1, 1]: a Cartesian displacement command
                                  (x 0.02 m per 0.04 s step) and a gripper
                                  command (-1 open, +1 closed)

One action is published for every accepted observation, in order. An
observation of the wrong length or with a non-finite value is rejected with a
warning and produces no action, so a downstream controller that times out on a
missing action fails safe instead of acting on a guess.

Parameters
    ``policy_path``  path to ``<name>.ts.pt`` (required)
    ``meta_path``    path to ``<name>.meta.json`` (default: next to the policy)
    ``num_threads``  torch intra-op threads (default 1)

What this node is not: a motion controller. The policy was trained on a
free-floating hand in MuJoCo, so its displacement command needs a reachability
and joint-limit check between this topic and any servo loop.
"""

from __future__ import annotations

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float32MultiArray, MultiArrayDimension

from grasp_policy_server.core import ObservationError, PolicyRunner

OBSERVATION_TOPIC = "grasp_policy/observation"
ACTION_TOPIC = "grasp_policy/action"

QOS = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=10,
                 reliability=ReliabilityPolicy.RELIABLE)


class PolicyServer(Node):
    def __init__(self) -> None:
        super().__init__("grasp_policy_server")
        policy_path = self.declare_parameter("policy_path", "").value
        meta_path = self.declare_parameter("meta_path", "").value or None
        num_threads = int(self.declare_parameter("num_threads", 1).value)

        self.runner = PolicyRunner(policy_path, meta_path, num_threads=num_threads)
        self.accepted = 0
        self.rejected = 0

        self.publisher = self.create_publisher(Float32MultiArray, ACTION_TOPIC, QOS)
        self.subscription = self.create_subscription(
            Float32MultiArray, OBSERVATION_TOPIC, self.on_observation, QOS)
        self.get_logger().info(
            "serving {} ({}-D observation -> {}-D action) on {} -> {}".format(
                policy_path, self.runner.observation_dim, self.runner.action_dim,
                OBSERVATION_TOPIC, ACTION_TOPIC))

    def on_observation(self, msg: Float32MultiArray) -> None:
        try:
            action = self.runner.act(msg.data)
        except ObservationError as exc:
            self.rejected += 1
            self.get_logger().warning(str(exc))
            return
        out = Float32MultiArray()
        out.layout.dim.append(MultiArrayDimension(
            label="action", size=len(action), stride=len(action)))
        out.data = action
        self.publisher.publish(out)
        self.accepted += 1


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = PolicyServer()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            # print, not the logger: after a SIGINT the context that owns
            # /rosout is already gone and the logger complains about it.
            print("served {} actions, rejected {} observations".format(
                node.accepted, node.rejected), flush=True)
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
