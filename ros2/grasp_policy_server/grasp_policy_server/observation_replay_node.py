"""Replay a recorded observation stream onto ``grasp_policy/observation``.

The stream is a JSON file written by ``scripts/record_observation_stream.py``:
one simulated MuJoCo episode, the observation the policy saw at every step and
the action it returned. This node publishes the observations in order at a
fixed rate and exits when the stream ends. It stands in for the part of a robot
cell that would assemble the observation from joint states, TF and perception,
which does not exist yet (``docs/ros2-policy-server.md``).

Parameters
    ``stream_path``                   the JSON stream (required)
    ``rate_hz``                       publishing rate (default 25, the 0.04 s
                                      control period the policy was trained at)
    ``wait_for_action_subscribers``   do not start until this many subscribers
                                      are listening on ``grasp_policy/action``
                                      (default 0); a test sets it to 1 so that
                                      no action is published before anyone
                                      is listening for it
    ``start_delay_s``                 settle time after discovery (default 1.0)
"""

from __future__ import annotations

import json
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, MultiArrayDimension

from grasp_policy_server.policy_server_node import ACTION_TOPIC, OBSERVATION_TOPIC, QOS


def load_stream(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        stream = json.load(fh)
    if not stream.get("observations"):
        raise ValueError("stream '{}' has no observations".format(path))
    return stream


class ObservationReplay(Node):
    def __init__(self) -> None:
        super().__init__("observation_replay")
        path = self.declare_parameter("stream_path", "").value
        self.rate_hz = float(self.declare_parameter("rate_hz", 25.0).value)
        self.wait_for_actions = int(
            self.declare_parameter("wait_for_action_subscribers", 0).value)
        self.start_delay = float(self.declare_parameter("start_delay_s", 1.0).value)
        self.observations = load_stream(path)["observations"]
        self.publisher = self.create_publisher(Float32MultiArray, OBSERVATION_TOPIC, QOS)
        self.get_logger().info("loaded {} observations from {}".format(
            len(self.observations), path))

    def wait_for_graph(self, timeout_s: float = 60.0) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if (self.publisher.get_subscription_count() >= 1
                    and self.count_subscribers(ACTION_TOPIC) >= self.wait_for_actions):
                time.sleep(self.start_delay)
                return True
            rclpy.spin_once(self, timeout_sec=0.1)
        return False

    def replay(self) -> int:
        period = 1.0 / self.rate_hz
        next_tick = time.monotonic()
        for row in self.observations:
            msg = Float32MultiArray()
            msg.layout.dim.append(MultiArrayDimension(
                label="observation", size=len(row), stride=len(row)))
            msg.data = [float(v) for v in row]
            self.publisher.publish(msg)
            next_tick += period
            time.sleep(max(0.0, next_tick - time.monotonic()))
        return len(self.observations)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = ObservationReplay()
        if not node.wait_for_graph():
            node.get_logger().error("nobody subscribed to {} within 60 s".format(
                OBSERVATION_TOPIC))
            raise SystemExit(1)
        sent = node.replay()
        node.get_logger().info("replay finished: {} observations published".format(sent))
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
