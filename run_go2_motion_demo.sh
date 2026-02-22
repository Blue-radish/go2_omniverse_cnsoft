#!/bin/bash
set -e

# ─────────────────────────────────────────────────────────
# 1. Activate Conda env and capture Python path
# ─────────────────────────────────────────────────────────
echo "[INFO] Activating Conda environment: paddle_env"
eval "$(conda shell.bash hook)"
conda activate paddle_env

PYTHON_PATH=$(which python)
echo "[INFO] Using Python interpreter: $PYTHON_PATH"

# ─────────────────────────────────────────────────────────
# 2. Source ROS 2 + workspace
# ─────────────────────────────────────────────────────────
echo "[INFO] Sourcing ROS 2 Humble setup"
source /opt/ros/humble/setup.bash

echo "[INFO] Rebuilding paddle_ros2 package"
cd ~/go2_omniverse_cnsoft/go2_omniverse_ws
colcon build --packages-select paddle_ros2 --symlink-install

echo "[INFO] Sourcing workspace setup"
source install/setup.bash

ENTRY_SCRIPT="install/paddle_ros2/lib/paddle_ros2/go2_motion_demo"

# ─────────────────────────────────────────────────────────
# 3. Patch shebang
# ─────────────────────────────────────────────────────────
if [ -f "$ENTRY_SCRIPT" ]; then
    echo "[INFO] Patching shebang in $ENTRY_SCRIPT"
    sed -i "1c\#!$PYTHON_PATH" "$ENTRY_SCRIPT"
else
    echo "[ERROR] Entry script not found: $ENTRY_SCRIPT"
    exit 1
fi

# ─────────────────────────────────────────────────────────
# 4. Preload libstdc++ and launch node
# ─────────────────────────────────────────────────────────
echo "[INFO] Preloading correct libstdc++"
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

echo "[INFO] Launching paddle_ros2 go2_motion_demo"
ros2 run paddle_ros2 go2_motion_demo
