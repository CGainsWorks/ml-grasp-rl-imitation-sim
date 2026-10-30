"""Start the policy server and optionally replay a recorded stream into it.

    ros2 launch grasp_policy_server policy_server.launch.py \
        policy_path:=experiments/policies/bcrl_medium_s0.ts.pt

    ros2 launch grasp_policy_server policy_server.launch.py \
        policy_path:=experiments/policies/bcrl_medium_s0.ts.pt \
        stream_path:=ros2/grasp_policy_server/test/data/replay_bcrl_medium_s0.json
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    policy_path = LaunchConfiguration("policy_path")
    stream_path = LaunchConfiguration("stream_path")
    return LaunchDescription([
        DeclareLaunchArgument("policy_path", description="exported <name>.ts.pt"),
        DeclareLaunchArgument("stream_path", default_value="",
                              description="optional recorded stream to replay"),
        Node(package="grasp_policy_server", executable="policy_server",
             name="grasp_policy_server", output="screen",
             parameters=[{"policy_path": policy_path}]),
        Node(package="grasp_policy_server", executable="observation_replay",
             name="observation_replay", output="screen",
             parameters=[{"stream_path": stream_path}],
             condition=IfCondition(PythonExpression(["'", stream_path, "' != ''"]))),
    ])
