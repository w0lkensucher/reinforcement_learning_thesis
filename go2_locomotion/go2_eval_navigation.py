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

from tqdm import tqdm
from  contextlib import redirect_stdout, redirect_stderr
from io import StringIO

import torch
import numpy as np
from datetime import datetime

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
    parser.add_argument("--record", action='store_true', default=False,
                        help="Record evaluation video")

    parser.add_argument("--obstacles", type=str, choices=["none", "sparse", "slalom", "corridor"], 
                        default="none", help="Type of obstacles to add during evaluation")
    parser.add_argument("--silent", action='store_true', help="Suppress detailed output")
    parser.add_argument("--dynamic_obstacles", action='store_true', help="Enable dynamic obstacles during evaluation")
    args = parser.parse_args()

    gs.init()
    
    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f"logs/{args.exp_name}/cfgs.pkl", "rb"))
    reward_cfg["reward_scales"] = {}
    env_cfg["terminate_on_collision"] = True
    
    # max_sim_step = int(env_cfg["episode_length_s"] * 30) 
    episode_length_s = 60
    max_sim_step = int(episode_length_s * 30)  # Assuming 30 steps per second

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

    if args.dynamic_obstacles:
        env_cfg["use_obstacles"] = True
        env_cfg["obstacle_density"] = 0.0
        env_cfg["terrain_size"] = [25.0, 25.0]
        env_cfg["obstacle_types"] = ["box"]
        env_cfg["obstacle_height_range"] = [0.05, 0.15]
        env_cfg["obstacle_width_range"] = [0.15, 0.3]
        env_cfg["obstacle_spacing_min"] = 2.0
        env_cfg["clear_radius"] = 3.0

    if args.silent:
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            env = Go2NavigationEnv(
                num_envs=1,
                env_cfg=env_cfg,
                obs_cfg=obs_cfg,
                reward_cfg=reward_cfg, 
                command_cfg=command_cfg,
                show_viewer=True,
                dynamic_obstacles=args.dynamic_obstacles,
            )
    else:
        env = Go2NavigationEnv(
            num_envs=1,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg, 
            command_cfg=command_cfg,
            show_viewer=True,
            dynamic_obstacles=args.dynamic_obstacles,
        )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    runner.load(resume_path)

    policy = runner.get_inference_policy(device=gs.device)

    obs, _ = env.reset()

    with torch.no_grad():
        if args.record:
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                env.cam.start_recording()

            print("🎥 Recording video...")

        with tqdm(total=max_sim_step, 
                desc="🤖 Evaluating Navigation", 
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
                file=sys.stdout,
                dynamic_ncols=True) as pbar:
            

            for step in range(max_sim_step):
                actions = policy(obs)
                obs, rews, dones, infos = env.step(actions)

                # Over-the-shoulder camera: behind and above the robot, looking forward
                base_pos = env.base_pos[0].cpu().numpy()
                base_yaw = env.base_radians[0, 2].cpu().item()

                cam_behind  = 2.5   # meters behind robot
                cam_above   = 1.5   # meters above robot
                look_ahead  = 2.0   # lookat point ahead of robot

                cam_x = base_pos[0] - cam_behind * np.cos(base_yaw)
                cam_y = base_pos[1] - cam_behind * np.sin(base_yaw)
                cam_z = base_pos[2] + cam_above

                look_x = base_pos[0] + look_ahead * np.cos(base_yaw)
                look_y = base_pos[1] + look_ahead * np.sin(base_yaw)
                look_z = base_pos[2] + 0.3  # slightly above ground level

                env.cam.set_pose(pos=(cam_x, cam_y, cam_z), lookat=(look_x, look_y, look_z))
                env.cam.render()

                reset_idx = dones.nonzero(as_tuple=False).squeeze(-1)
                if (len(reset_idx) > 0):
                    env.reset_idx(reset_idx)
                pbar.update(1)

        if args.record:
            print("🎥 Stopping recording...")
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                env.cam.stop_recording(save_to_filename=f"videos/{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.exp_name}.mp4",fps=30)

if __name__ == "__main__":
    main()
    