#!/bin/bash
set -e

# ─────────────────────────────────────────────────────────
# 1. Activate Conda env and discover the Python path
# ─────────────────────────────────────────────────────────
echo "[INFO] Activating Conda environment: paddle_env"
eval "$(conda shell.bash hook)"
conda activate paddl

PYTHON_PATH=$(which python)  # e.g. /home/ziyueg/miniconda3/envs/paddle_env/bin/python
echo "[INFO] Using Python interpreter: $PYTHON_PATH"

# ─────────────────────────────────────────────────────────
# 2. Source ROS 2 and your workspace
# ─────────────────────────────────────────────────────────
echo "[INFO] Sourcing ROS 2 Humble setup"
source /opt/ros/humble/setup.bash

echo "[INFO] Rebuilding paddle_ros2 package"
cd ~/go2_omniverse_cnsoft/go2_omniverse_ws
colcon build --packages-select paddle_ros2 --symlink-install

echo "[INFO] Sourcing workspace setup"
source install/setup.bash

# Path to the console-script that colcon just produced
ENTRY_SCRIPT="install/paddle_ros2/lib/paddle_ros2/ernie_llm_converter"

# ─────────────────────────────────────────────────────────
# 3. Patch the shebang if the file exists
# ─────────────────────────────────────────────────────────
if [ -f "$ENTRY_SCRIPT" ]; then
    echo "[INFO] Patching shebang in $ENTRY_SCRIPT"
    sed -i "1c\#!$PYTHON_PATH" "$ENTRY_SCRIPT"
else
    echo "[ERROR] Entry script not found: $ENTRY_SCRIPT"
    exit 1
fi

# ─────────────────────────────────────────────────────────
# 4. (Optional) proxy toggle
# ─────────────────────────────────────────────────────────
read -p "Do you want to set the ALL_PROXY environment variable? [Y/n]: " set_proxy
set_proxy=${set_proxy,,}  # force to lowercase

if [[ "$set_proxy" == "y" || "$set_proxy" == "" ]]; then
    echo "[INFO] Setting proxy environment variables"
    export ALL_PROXY="socks5://127.0.0.1:7891"
    export all_proxy="$ALL_PROXY"
    export HTTP_PROXY="$ALL_PROXY"
    export http_proxy="$ALL_PROXY"
    export HTTPS_PROXY="$ALL_PROXY"
    export https_proxy="$ALL_PROXY"
else
    echo "[INFO] Skipping proxy setup"
fi

# ─────────────────────────────────────────────────────────
# 5. Preload libstdc++ and launch the node
# ─────────────────────────────────────────────────────────
echo "[INFO] Preloading correct libstdc++"
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

echo "[INFO] Launching paddle_ros2 ernie_llm_converter"
ros2 run paddle_ros2 ernie_llm_converter

