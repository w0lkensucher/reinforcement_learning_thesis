import os
if os.name != 'nt':
    os.environ['SETUPTOOLS_USE_DISTUTILS'] = 'local'

    import sys
    if 'distutils' in sys.modules:
        del sys.modules['distutils']

    import importlib
    distutils_core = importlib.import_module('distutils.core')
    print("Distutils core loaded from:", distutils_core.__file__)
    
import argparse
import pickle
from importlib import metadata

import torch
import numpy as np

try:
    try:
        if metadata.version("rsl-rl"):
            raise ImportError
    except metadata.PackageNotFoundError:
        if metadata.version("rsl-rl-lib") != "2.2.4":
            raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please uninstall 'rsl_rl' and install 'rsl-rl-lib==2.2.4'.") from e
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from go2_env import Go2Env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2-walking")
    parser.add_argument("--ckpt", type=int, default=100)
    parser.add_argument("--obstacles", type=str, choices=["none", "sparse", "slalom", "corridor"], 
                        default="none", help="Type of obstacles to add during evaluation")
    parser.add_argument("--speed_multiplier", type=float, default=1.0,
                        help="Multiply command velocities by this factor")
    args = parser.parse_args()

    gs.init()

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f"logs/{args.exp_name}/cfgs.pkl", "rb"))
    reward_cfg["reward_scales"] = {}

    # IMPORTANT: Override obstacle settings in env_cfg BEFORE creating environment
    if args.obstacles != "none":
        print(f"🏗️  Enabling obstacles for evaluation: {args.obstacles}")
        # Enable obstacles in the environment configuration
        env_cfg["use_obstacles"] = True
        
        # Set default obstacle configuration (safe values)
        obstacle_defaults = {
            "obstacle_density": 0.02,
            "obstacle_types": ["box", "cylinder"],
            "obstacle_height_range": [0.05, 0.15],
            "obstacle_width_range": [0.15, 0.3],
            "obstacle_spacing_min": 2.0,
            "terrain_size": [25.0, 25.0],
            "clear_radius": 3.0,
        }
        
        # Apply defaults first
        for key, value in obstacle_defaults.items():
            env_cfg[key] = value
        
        # Then override specific settings based on obstacle type
        if args.obstacles == "sparse":
            env_cfg["obstacle_density"] = 0.015  # Even fewer obstacles
        elif args.obstacles == "corridor":
            env_cfg["obstacle_density"] = 0.04
            env_cfg["obstacle_types"] = ["box"]
            env_cfg["obstacle_height_range"] = [0.1, 0.2]
            env_cfg["obstacle_width_range"] = [0.3, 0.6]
            env_cfg["obstacle_spacing_min"] = 1.5
            env_cfg["clear_radius"] = 2.5
        elif args.obstacles == "slalom":
            env_cfg["obstacle_density"] = 0.025
            env_cfg["obstacle_types"] = ["cylinder"]
            env_cfg["obstacle_height_range"] = [0.1, 0.2]
            env_cfg["obstacle_width_range"] = [0.2, 0.3]
            env_cfg["obstacle_spacing_min"] = 2.0
    else:
        # Ensure obstacles are explicitly disabled
        env_cfg["use_obstacles"] = False

    # Apply speed multiplier if specified
    if args.speed_multiplier != 1.0:
        print(f"🚀 Applying speed multiplier: {args.speed_multiplier}x")
        orig_x_range = command_cfg["lin_vel_x_range"]
        orig_y_range = command_cfg["lin_vel_y_range"]
        orig_ang_range = command_cfg["ang_vel_range"]
        
        command_cfg["lin_vel_x_range"] = [x * args.speed_multiplier for x in orig_x_range]
        command_cfg["lin_vel_y_range"] = [x * args.speed_multiplier for x in orig_y_range]
        command_cfg["ang_vel_range"] = [x * args.speed_multiplier for x in orig_ang_range]
        
        print(f"  Forward: {orig_x_range} -> {command_cfg['lin_vel_x_range']}")

    env = Go2Env(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=gs.device)

    print(f"🎯 Starting evaluation:")
    print(f"  Experiment: {args.exp_name}")
    print(f"  Checkpoint: {args.ckpt}")
    print(f"  Obstacles: {args.obstacles}")
    print(f"  Speed multiplier: {args.speed_multiplier}x")
    print(f"  Press Ctrl+C to stop\n")

    obs, _ = env.reset()
    step_count = 0
    with torch.no_grad():
        try:
            while True:
                actions = policy(obs)
                obs, rews, dones, infos = env.step(actions)
                step_count += 1
                
                # Print periodic updates
                if step_count % 1000 == 0:
                    print(f"Step {step_count}: Running evaluation...")
                    
        except KeyboardInterrupt:
            print(f"\n✅ Evaluation stopped after {step_count} steps")


if __name__ == "__main__":
    main()

"""
# evaluation examples

# Basic evaluation (no obstacles)
python examples/locomotion/go2_eval.py -e go2-walking --ckpt 100

# Evaluation with sparse obstacles
python examples/locomotion/go2_eval.py -e go2-walking --ckpt 100 --obstacles sparse

# Slalom course evaluation
python examples/locomotion/go2_eval.py -e go2-walking --ckpt 100 --obstacles slalom

# Corridor navigation test
python examples/locomotion/go2_eval.py -e go2-walking --ckpt 100 --obstacles corridor

# High-speed obstacle navigation
python examples/locomotion/go2_eval.py -e go2-walking --ckpt 100 --obstacles sparse --speed_multiplier 1.5
"""
