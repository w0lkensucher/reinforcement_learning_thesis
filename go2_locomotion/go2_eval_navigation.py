import sys
import os
if sys.version_info < (3, 12):
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

from go2_env_navigation import Go2NavigationEnv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2-navigation")
    parser.add_argument("--ckpt", type=int, default=100)

    parser.add_argument("--obstacles", type=str, choices=["none", "sparse", "slalom", "corridor"], 
                        default="none", help="Type of obstacles to add during evaluation")
    
    args = parser.parse_args()

    gs.init()
    
    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f"logs/{args.exp_name}/cfgs.pkl", "rb"))
    reward_cfg["reward_scales"] = {}

    if args.obstacles != "none":
        env_cfg["use_obstacles"] = True

        obstacle_defaults = {
            "obstacle_density": 0.02,
            "obstacle_types": ["box", "cylinder"],
            "obstacle_height_range": [0.05, 0.15],
            "obstacle_width_range": [0.15, 0.3],
            "obstacle_spacing_min": 2.0,
            "terrain_size": [25.0, 25.0],
            "clear_radius": 3.0,
        }

        for key, value in obstacle_defaults.items():
            env_cfg[key] = value

        if args.obstacles == "sparse":
            env_cfg["obstacle_density"] = 0.01
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
        env_cfg["use_obstacles"] = False

    env = Go2NavigationEnv(
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

    obs, _ = env.reset()

    with torch.no_grad():
        while True:
            actions = policy(obs)
            obs, rews, dones, infos = env.step(actions)


if __name__ == "__main__":
    main()
    