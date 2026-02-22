#!/bin/bash
set -e

echo "[INFO] Activating Conda environment: paddl"
eval "$(conda shell.bash hook)"
conda activate paddl

PYTHON_PATH=$(which python)
ENTRY_SCRIPT="$HOME/go2_omniverse_cnsoft/go2_omniverse_ws/install/paddle_ros2/lib/paddle_ros2/ocr_node"

echo "[INFO] Sourcing ROS2 Humble setup"
source /opt/ros/humble/setup.bash

echo "[INFO] Rebuilding paddle_ros2 package"
cd ~/go2_omniverse_cnsoft/go2_omniverse_ws
colcon build --packages-select paddle_ros2 --symlink-install

# Wait for build to complete and ensure the entry script exists
if [ -f "$ENTRY_SCRIPT" ]; then
    echo "[INFO] Patching ocr_node shebang to use Conda Python: $PYTHON_PATH"
    sed -i "1c\#!$PYTHON_PATH" "$ENTRY_SCRIPT"
else
    echo "[ERROR] ocr_node entry script not found at $ENTRY_SCRIPT"
    exit 1
fi

echo "[INFO] Sourcing workspace setup again"
source install/setup.bash

echo "[INFO] Preloading correct libstdc++"
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

echo "[INFO] Launching paddle_ros2 ocr_node"
ros2 run paddle_ros2 ocr_node

