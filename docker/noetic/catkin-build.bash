#!/usr/bin/env bash
set -euo pipefail

source /opt/ros/noetic/setup.bash

cd /ws
catkin init
catkin config \
    --extend /opt/ros/noetic \
    --merge-devel \
    --cmake-args \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_FLAGS=-fdiagnostics-color \
        -DEIGEN_FROM_SYSTEM=ON \
        -DBUILD_TESTS=OFF \
        -DBUILD_UNITY_BRIDGE_TESTS=OFF
catkin build --no-status --summarize
