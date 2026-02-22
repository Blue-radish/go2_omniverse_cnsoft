"""
Run an RSL-RL checkpoint inside Isaac Sim, stream front-camera images
to ROS 2, and accept /robot{i}/cmd_vel (Twist) to bias base motion.

Tested with:
  • Isaac Sim 4.2.1 + Isaac-Lab 1.4.1
  • ROS 2 Humble
  • Unitree Go2 and G1 tasks
"""

from __future__ import annotations

# ───────────────────── Launch Isaac app ────────────────────────────
import argparse, os, time, threading

from omni.isaac.lab.app import AppLauncher
import cli_args

# CLI arguments
parser = argparse.ArgumentParser(description="Play a trained RL agent.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str,
                    default="Isaac-Velocity-Rough-Unitree-Go2-v0")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--custom_env", type=str, default="office")
parser.add_argument("--robot", type=str, default="go2")
parser.add_argument("--terrain", type=str, default="rough")
parser.add_argument("--robot_amount", type=int, default=1)

cli_args.add_rsl_rl_args(parser)           # extra RSL-RL flags
AppLauncher.add_app_launcher_args(parser)  # kit flags
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ───────────────────── Omni extension toggles ─────────────────────
import omni
ext_manager = omni.kit.app.get_app().get_extension_manager()
ext_manager.set_extension_enabled_immediate("omni.isaac.ros2_bridge", True)

# ───────────────────── Standard imports after app ─────────────────
import gymnasium as gym
import torch, carb, omni.appwindow, omni.isaac.lab.sim as sim_utils
from rsl_rl.runners import OnPolicyRunner
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
)
from omni.isaac.lab_tasks.utils.parse_cfg import get_checkpoint_path

import rclpy
from ros2 import RobotBaseNode, add_camera, add_rtx_lidar, pub_robo_data_ros2
from geometry_msgs.msg import Twist

from agent_cfg import unitree_go2_agent_cfg, unitree_g1_agent_cfg
from custom_rl_env import UnitreeGo2CustomEnvCfg, G1RoughEnvCfg
import custom_rl_env
from omnigraph import create_front_cam_omnigraph

# ───────────────────── Keyboard tele-op callback ──────────────────
def sub_keyboard_event(event, *args, **kwargs) -> bool:
    """
    Map WASDQE(IKJLUO) keys to base_command dict so we can still steer
    each robot manually.
    """
    if len(custom_rl_env.base_command) > 0:
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name == 'W':
                custom_rl_env.base_command["0"] = [1, 0, 0]
            if event.input.name == 'S':
                custom_rl_env.base_command["0"] = [-1, 0, 0]
            if event.input.name == 'A':
                custom_rl_env.base_command["0"] = [0, 1, 0]
            if event.input.name == 'D':
                custom_rl_env.base_command["0"] = [0, -1, 0]
            if event.input.name == 'Q':
                custom_rl_env.base_command["0"] = [0, 0, 1]
            if event.input.name == 'E':
                custom_rl_env.base_command["0"] = [0, 0, -1]

            if len(custom_rl_env.base_command) > 1:
                if event.input.name == 'I':
                    custom_rl_env.base_command["1"] = [1, 0, 0]
                if event.input.name == 'K':
                    custom_rl_env.base_command["1"] = [-1, 0, 0]
                if event.input.name == 'J':
                    custom_rl_env.base_command["1"] = [0, 1, 0]
                if event.input.name == 'L':
                    custom_rl_env.base_command["1"] = [0, -1, 0]
                if event.input.name == 'U':
                    custom_rl_env.base_command["1"] = [0, 0, 1]
                if event.input.name == 'O':
                    custom_rl_env.base_command["1"] = [0, 0, -1]

        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            for i in range(len(custom_rl_env.base_command)):
                custom_rl_env.base_command[str(i)] = [0, 0, 0]
    return True

# ───────────────────── Custom environment loader ──────────────────
def setup_custom_env():
    try:
        if (args_cli.custom_env == "warehouse" and args_cli.terrain == 'flat'):
            cfg_scene = sim_utils.UsdFileCfg(usd_path="./envs/warehouse.usd")
            cfg_scene.func("/World/warehouse", cfg_scene, translation=(0.0, 0.0, 0.0))

        if (args_cli.custom_env == "office" and args_cli.terrain == 'flat'):
            cfg_scene = sim_utils.UsdFileCfg(usd_path="./envs/office.usd")
            cfg_scene.func("/World/office", cfg_scene, translation=(0.0, 0.0, 0.0))

        if (args_cli.custom_env == "cnsoft" and args_cli.terrain == 'flat'):
            cfg_scene = sim_utils.UsdFileCfg(usd_path="./envs/cnsoft.usd")
            cfg_scene.func("/World/cnsoft", cfg_scene, translation=(0.0, 0.0, 0.0))

        if (args_cli.custom_env == "football" and args_cli.terrain == 'flat'):
            cfg_scene = sim_utils.UsdFileCfg(usd_path="./envs/football.usd")
            cfg_scene.func("/World/football", cfg_scene, translation=(0.0, 0.0, 0.0))

    except:
        print(
            "Error loading custom environment. You should download custom envs folder from: https://drive.google.com/drive/folders/1vVGuO1KIX1K6mD6mBHDZGm9nk2vaRyj3?usp=sharing")

# ───────────────────── ROS 2 cmd_vel subscriber glue ──────────────
base_command = {}  # {str(idx): [x, y, z]}

def _cmd_cb(msg: Twist, idx: str):
    base_command[idx] = [msg.linear.x, msg.linear.y, msg.angular.z]

def init_cmd_subscribers(num_envs: int):
    """
    Create /robot{i}/cmd_vel subscribers in a background thread.
    If rclpy is already initialised, reuse that context instead of
    calling rclpy.init() again.
    """
    try:
        rclpy.init(args=None)
    except RuntimeError:
        # already initialised – safe to continue
        pass

    node = rclpy.create_node("isaac_cmd_router")

    for i in range(num_envs):
        base_command[str(i)] = [0.0, 0.0, 0.0]
        node.create_subscription(
            Twist,
            f"/robot{i}/cmd_vel",
            lambda m, i=i: _cmd_cb(m, str(i)),
            10,
        )

    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    print("[ROS2] cmd_router spinning in background")

# ───────────────────── Main simulation function ───────────────────
def specify_cmd_for_robots(num_envs: int):
    for i in range(num_envs):
        custom_rl_env.base_command[str(i)] = [0, 0, 0]

def run_sim():
    # keyboard hook
    _input = carb.input.acquire_input_interface()
    _keyboard = omni.appwindow.get_default_app_window().get_keyboard()
    _input.subscribe_to_keyboard_events(_keyboard, sub_keyboard_event)

    # env cfg
    env_cfg = UnitreeGo2CustomEnvCfg()
    if args_cli.robot == "g1":
        env_cfg = G1RoughEnvCfg()
    env_cfg.scene.num_envs = args_cli.robot_amount

    # create camera graphs
    for i in range(env_cfg.scene.num_envs):
        create_front_cam_omnigraph(i)
    specify_cmd_for_robots(env_cfg.scene.num_envs)

    # agent cfg
    agent_cfg: RslRlOnPolicyRunnerCfg = unitree_go2_agent_cfg
    if args_cli.robot == "g1":
        agent_cfg = unitree_g1_agent_cfg

    # create isaac env and RSL-RL wrapper
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    # load trained policy
    log_root = os.path.abspath(os.path.join("logs", "rsl_rl",
                                            agent_cfg["experiment_name"]))
    resume = get_checkpoint_path(log_root,
                                 agent_cfg["load_run"],
                                 agent_cfg["load_checkpoint"])
    ppo_runner = OnPolicyRunner(env, agent_cfg,
                                log_dir=None, device=agent_cfg["device"])
    ppo_runner.load(resume)
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # ROS 2 publishers (cameras, lidar)
    base_node = None
    try:
        rclpy.init()
        base_node = RobotBaseNode(env_cfg.scene.num_envs)
    except RuntimeError:
        pass  # already initialised by cmd_router

    init_cmd_subscribers(env_cfg.scene.num_envs)
    annotator_lst = add_rtx_lidar(env_cfg.scene.num_envs, args_cli.robot, False)
    add_camera(env_cfg.scene.num_envs, args_cli.robot)
    setup_custom_env()

    # main loop
    obs, _ = env.get_observations()
    start_time = time.time()

    while simulation_app.is_running():
        # inject latest /cmd_vel into base_command
        for k, v in base_command.items():
            custom_rl_env.base_command[k] = v

        # RL inference & step
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)

        pub_robo_data_ros2(args_cli.robot,
                           env_cfg.scene.num_envs,
                           base_node, env, annotator_lst, start_time)

    env.close()

# ───────────────────── Entrypoint ─────────────────────────────────
if __name__ == "__main__":
    run_sim()
