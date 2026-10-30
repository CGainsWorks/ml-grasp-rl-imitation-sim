"""Launch test: a recorded observation stream through the live node.

Starts ``policy_server`` on the committed TorchScript policy and
``observation_replay`` on a stream recorded from one simulated MuJoCo episode
(``scripts/record_observation_stream.py``). The test subscribes to
``grasp_policy/action`` and requires one action per observation, in order,
matching what the policy returned in simulation to 1e-5. It then sends a
malformed observation and requires the node to reject it without publishing.

    launch_test ros2/grasp_policy_server/test/test_policy_server_launch.py
"""

from __future__ import annotations

import json
import os
import time
import unittest

import launch
import launch_ros.actions
import launch_testing
import launch_testing.actions
import launch_testing.asserts
import pytest
import rclpy
from std_msgs.msg import Float32MultiArray

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
POLICY = os.path.join(REPO, "experiments", "policies", "bcrl_medium_s0.ts.pt")
STREAM = os.path.join(HERE, "data", "replay_bcrl_medium_s0.json")
TOLERANCE = 1e-5


@pytest.mark.launch_test
def generate_test_description():
    server = launch_ros.actions.Node(
        package="grasp_policy_server", executable="policy_server",
        name="grasp_policy_server", output="screen",
        parameters=[{"policy_path": POLICY}])
    replay = launch_ros.actions.Node(
        package="grasp_policy_server", executable="observation_replay",
        name="observation_replay", output="screen",
        parameters=[{"stream_path": STREAM, "rate_hz": 50.0,
                     "wait_for_action_subscribers": 1, "start_delay_s": 2.0}])
    description = launch.LaunchDescription([
        server, replay, launch_testing.actions.ReadyToTest()])
    return description, {"server": server, "replay": replay}


class TestReplayedStream(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rclpy.init()
        cls.node = rclpy.create_node("policy_server_test")
        with open(STREAM, "r", encoding="utf-8") as fh:
            cls.stream = json.load(fh)

    @classmethod
    def tearDownClass(cls):
        cls.node.destroy_node()
        rclpy.shutdown()

    def test_1_actions_match_the_simulated_episode(self, proc_output, replay):
        from grasp_policy_server.policy_server_node import ACTION_TOPIC, QOS

        received = []
        sub = self.node.create_subscription(
            Float32MultiArray, ACTION_TOPIC, lambda m: received.append(list(m.data)), QOS)
        expected = self.stream["actions"]
        deadline = time.monotonic() + 90.0
        try:
            while len(received) < len(expected) and time.monotonic() < deadline:
                rclpy.spin_once(self.node, timeout_sec=0.1)
            # Anything beyond the stream would be a duplicate or a stray.
            settle = time.monotonic() + 1.0
            while time.monotonic() < settle:
                rclpy.spin_once(self.node, timeout_sec=0.1)
        finally:
            self.node.destroy_subscription(sub)

        self.assertEqual(len(received), len(expected),
                         "one action per replayed observation")
        worst = max(abs(a - b) for got, want in zip(received, expected)
                    for a, b in zip(got, want))
        self.assertLess(worst, TOLERANCE)
        proc_output.assertWaitFor("replay finished: {} observations published".format(
            len(expected)), process=replay, timeout=10)

    def test_2_malformed_observation_is_rejected(self, proc_output, server):
        from grasp_policy_server.policy_server_node import (
            ACTION_TOPIC, OBSERVATION_TOPIC, QOS)

        received = []
        sub = self.node.create_subscription(
            Float32MultiArray, ACTION_TOPIC, lambda m: received.append(m), QOS)
        pub = self.node.create_publisher(Float32MultiArray, OBSERVATION_TOPIC, QOS)
        try:
            deadline = time.monotonic() + 20.0
            while pub.get_subscription_count() < 1 and time.monotonic() < deadline:
                rclpy.spin_once(self.node, timeout_sec=0.1)
            time.sleep(1.0)
            msg = Float32MultiArray()
            msg.data = [0.0] * 31
            pub.publish(msg)
            proc_output.assertWaitFor("rejected observation of length 31",
                                      process=server, timeout=10)
            settle = time.monotonic() + 1.0
            while time.monotonic() < settle:
                rclpy.spin_once(self.node, timeout_sec=0.1)
            self.assertEqual(received, [], "a rejected observation must not produce an action")
        finally:
            self.node.destroy_publisher(pub)
            self.node.destroy_subscription(sub)


@launch_testing.post_shutdown_test()
class TestShutdown(unittest.TestCase):

    def test_processes_exit_cleanly(self, proc_info):
        launch_testing.asserts.assertExitCodes(proc_info)
