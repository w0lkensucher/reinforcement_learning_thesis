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
import shutil
import csv
import torch
import numpy as np
from importlib import metadata

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


def get_navigation_train_cfg(exp_name, max_iterations):
    return {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.14112594271747939,
            "desired_kl": 0.01,
            "entropy_coef": 0.09847227294970444,
            "gamma": 0.9821749211289272,
            "lam": 0.95,
            "learning_rate": 9.603951203562604e-05,
            "max_grad_norm": 0.5,
            "num_learning_epochs": 8,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 2.5805037983234484,
        },
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "init_noise_std": 1.0,
            "class_name": "ActorCritic",
        },
        "runner": {
            "experiment_name": exp_name,
            "max_iterations": max_iterations,
            "log_interval": 1,
            "record_interval": 50,
            "resume": False,
        },
        "num_steps_per_env": 24,
        "save_interval": 100,
        "seed": 1,
    }


def get_navigation_cfgs():
    env_cfg = {
        "num_actions": 12,
        "default_joint_angles": {  # [rad]
            "FL_hip_joint": 0.0,
            "FR_hip_joint": 0.0,
            "RL_hip_joint": 0.0,
            "RR_hip_joint": 0.0,
            "FL_thigh_joint": 0.8,
            "FR_thigh_joint": 0.8,
            "RL_thigh_joint": 1.0,
            "RR_thigh_joint": 1.0,
            "FL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,
            "RL_calf_joint": -1.5,
            "RR_calf_joint": -1.5,
        },
        "joint_names": [
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        ],
        # PD
        "kp": 20.0,
        "kd": 0.5,
        "termination_if_roll_greater_than": 45,  # degrees
        "termination_if_pitch_greater_than": 45,
        "base_init_pos": [0.0, 0.0, 0.42],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 25.0,
        "resampling_time_s": 6.0,
        "action_scale": 0.25,
        "simulate_action_latency": True,
        "clip_actions": 1.0,
        
        # NAVIGATION-SPECIFIC CONFIG
        'use_obstacles': True,
        'obstacle_density': 0.02,
        'obstacle_types': ['box', 'cylinder'],
        'obstacle_height_range': [0.05, 0.15],
        'obstacle_width_range': [0.1, 0.3],
        'obstacle_spacing_min': 1.5,
        'terrain_size': [30.0, 30.0],
        'clear_radius': 3.0,
        'jump_height_threshold': 0.08,  # Below this = low obstacle (jump)
    }
    
    obs_cfg = {
        "num_obs": 47,  # Base (45) + obstacle signals (2) + distances (2)
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
    }
    
    reward_cfg = {
        "jump_height_threshold": 0.08,
        "base_height_target": 0.42,
        "reward_scales": {
            # Navigation-focused rewards
            "tracking_lin_vel": 2.0,
            "tracking_ang_vel": 1.5,
            'forward_movement': 1.0,
            "straight_walk_when_clear": 1.0,
            "obstacle_avoidance": 2.0,
            "adaptive_base_height": 1.5,
            "orientation_stability": 2.0,
            "angular_velocity_stability": 1.5,
            "upright_posture": 1.0,
            "landing_stability": 0.5,
            "ground_clearance": 0.5,
            "landing_stability": 1.0,
            
            # Base rewards (reduced for navigation)
            "lin_vel_z": 1.0,
            "action_rate": 0.5,
            "similar_to_default": 0.2,
        },
    }
    
    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [0.3, 1.2],   # Faster for navigation
        "lin_vel_y_range": [-0.3, 0.3],  # Allow lateral movement
        "ang_vel_range": [-0.5, 0.5],    # Allow turning
    }
    
    return env_cfg, obs_cfg, reward_cfg, command_cfg


def get_resume_train_cfg(exp_name, max_iterations, resume_path=None):
    cfg = get_navigation_train_cfg(exp_name, max_iterations)
    cfg["runner"].update({
        "resume": True,
        "load_run": -1,           # Load latest run
        "checkpoint": -1,         # Load latest checkpoint
        "resume_path": resume_path,  # Or specify exact path
    })
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2-navigation")
    parser.add_argument("-B", "--num_envs", type=int, default=2048)
    parser.add_argument("--max_iterations", type=int, default=500)
    parser.add_argument("--uneven_terrain", action='store_true')
    parser.add_argument("--wind_force", action='store_true')
    args = parser.parse_args()

    gs.init(logging_level="warning")
    
    log_dir = f"logs/{args.exp_name}"
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    env_cfg, obs_cfg, reward_cfg, command_cfg = get_navigation_cfgs()
    train_cfg = get_navigation_train_cfg(args.exp_name, args.max_iterations)

    # Save configs
    pickle.dump([env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
                open(f"{log_dir}/cfgs.pkl", "wb"))

    # Create NAVIGATION environment
    env = Go2NavigationEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        uneven_terrain=args.uneven_terrain,
        wind_force=args.wind_force
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()

# Usage:
# python go2_train_navigation.py -e go2-nav --num_envs 2048 --max_iterations 500
# python go2_train_navigation.py -e go2-nav-terrain --uneven_terrain --wind_force