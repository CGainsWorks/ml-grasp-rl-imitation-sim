#!/usr/bin/env bash
# Build and test the ROS 2 policy server inside the official Humble image,
# the same steps the `ros2` CI job runs.
#
#   make ros-test
#   # which is:
#   docker run --rm -v "$PWD":/src:ro ros:humble-ros-base-jammy bash /src/scripts/ros_test.sh
#
# The checkout is mounted read-only and copied into the container, so nothing
# the build writes lands in the working tree with the container's ownership.
set -euo pipefail

SRC="${SRC:-/src}"
WS=/ws

apt-get update -qq
apt-get install -y -qq --no-install-recommends \
    python3-pip python3-pytest python3-colcon-common-extensions >/dev/null
# --no-deps: see the comment on the same line in .github/workflows/ci.yml.
pip install -q --no-deps torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
pip install -q filelock "typing-extensions>=4.10" "sympy>=1.13.3" \
    "networkx>=2.5.1" jinja2 "fsspec>=0.8.5"

mkdir -p "$WS/experiments"
cp -r "$SRC/ros2" "$WS/ros2"
cp -r "$SRC/experiments/policies" "$WS/experiments/policies"
cd "$WS"

set +u   # the ROS setup scripts read unset variables
. /opt/ros/humble/setup.bash
colcon build --base-paths ros2
. install/setup.bash
set -u
colcon test --base-paths ros2 --event-handlers console_direct+
colcon test-result --verbose
