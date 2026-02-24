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
import shutil
import csv
import torch
import numpy as np
import re
from importlib import metadata
from datetime import datetime
from tqdm import tqdm

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
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,  # Overridden per curriculum stage in main()
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.001,
            "max_grad_norm": 0.5,
            "num_learning_epochs": 8,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "init_noise_std": 0.5,  # Lower for standing - converges faster, less initial chaos
            "class_name": "ActorCritic",
        },
        "runner": {
            "experiment_name": exp_name,
            "max_iterations": max_iterations,
            "log_interval": 1,
            "record_interval": 50,
            "resume": False,
        },
        "num_steps_per_env": 48,  # Longer rollouts = cleaner value estimates = less noisy gradients
        "save_interval": 10,
        "empirical_normalization": None,
        "seed": 8, # set to different seeds for multiple runs
    }


def get_navigation_cfgs(curriculum_stage=1):
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
        "termination_if_roll_greater_than": 7,  # degrees
        "termination_if_pitch_greater_than": 7,
        "base_init_pos": [0.0, 0.0, 0.42],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 60.0,
        "resampling_time_s": 6.0,
        "action_scale": 0.25,
        "simulate_action_latency": True,
        "clip_actions": 1.0,
        
        # NAVIGATION-SPECIFIC CONFIG
        'use_obstacles': True,
        'obstacle_density': 0.01,
        'obstacle_types': ['box', 'cylinder'],
        'obstacle_height_range': [0.05, 0.15],
        'obstacle_width_range': [0.1, 0.3],
        'obstacle_spacing_min': 1.5,
        'terrain_size': [30.0, 30.0],
        'clear_radius': 2.0,
        'jump_height_threshold': 0.08,  # Below this = low obstacle (jump)
    }
    
    obs_cfg = {
        "num_obs": 53,  # Base (45) + obstacle signals (2)
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
    }
    
    reward_cfg = {
        "reward_scales":{key: 0.0 for key in [
            "tracking_lin_vel",
            "tracking_ang_vel",
            "forward_movement",
            "straight_walk_when_clear",
            "obstacle_avoidance",
            "height",
            "upright",
            "landing_stability",
            "jump_clearance",
            "lin_vel_z",
            "action_rate",
            "similar_to_default",
            "symmetry",
            "target_proximity",
            "target_reached",
            "target_progress",
            "goal_heading",
            "keep_moving",
            "blocker_clearance",
        ]},
        "base_height_target": 0.42,
    }
    
    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [0.3, 1.2],   # Faster for navigation
        "lin_vel_y_range": [-0.3, 0.3],  # Allow lateral movement
        "ang_vel_range": [-0.5, 0.5],    # Allow turning
    }


    if curriculum_stage == 1:
        # Stage 1: Standing safely
        # Zero commands - robot should stand perfectly still, no velocity tracking needed
        command_cfg["lin_vel_x_range"] = [0.0, 0.0]
        command_cfg["lin_vel_y_range"] = [0.0, 0.0]
        command_cfg["ang_vel_range"] = [0.0, 0.0]

        reward_cfg["tracking_sigma"] = 0.5
        reward_cfg["reward_scales"]["upright"] = 5.0             # Primary signal: stay upright
        reward_cfg["reward_scales"]["height"] = 3.0              # Maintain proper height

        reward_cfg["reward_scales"]["tracking_lin_vel"] = 0.0    # DISABLED: commands are zero, no gradient
        reward_cfg["reward_scales"]["tracking_ang_vel"] = 0.0    # DISABLED: commands are zero, no gradient
        reward_cfg["reward_scales"]["action_rate"] = -0.05       # Small smoothness penalty
        reward_cfg["reward_scales"]["similar_to_default"] = -0.1 # Small pose penalty, not too strong
        reward_cfg["reward_scales"]["lin_vel_z"] = -0.2          # Penalize bouncing
        env_cfg["episode_length_s"] = 20.0
        env_cfg["use_obstacles"] = False

        # Relaxed termination for standing with balance adjustments
        env_cfg["termination_if_pitch_greater_than"] = 12
        env_cfg["termination_if_roll_greater_than"] = 12
        env_cfg["termination_if_base_height_lower_than"] = 0.25

    elif curriculum_stage == 2:
        # Stage 2: Walking
        command_cfg["lin_vel_x_range"] = [0.3, 0.7]    # Straight forward walking
        command_cfg["lin_vel_y_range"] = [0.0, 0.0]    # No lateral - prevents rightward drift
        command_cfg["ang_vel_range"] = [0.0, 0.0]      # No turning - prevents heading drift

        reward_cfg["tracking_sigma"] = 0.25             # Tight tracking - standing still with 0.3 cmd gets heavily penalized

        # WALKING is PRIMARY - flip the balance vs standing config
        reward_cfg["reward_scales"]["tracking_lin_vel"] = 4.0      # Dominant reward - must move forward
        reward_cfg["reward_scales"]["tracking_ang_vel"] = 0.0      # Disabled - no turning commands

        # Stability is SECONDARY - just enough to prevent falling
        reward_cfg["reward_scales"]["upright"] = 2.5               # Keep robot upright
        reward_cfg["reward_scales"]["height"] = 1.5                # Maintain proper height

        # Explicit penalty for standing still - robot cannot reward-hack by not moving
        reward_cfg["reward_scales"]["forward_movement"] = 1.5      # Penalize zero forward velocity

        # Movement quality - stronger to enforce stable, symmetric gait
        reward_cfg["reward_scales"]["action_rate"] = -0.15         # Stronger smoothness - reduces jerky/unstable motion
        reward_cfg["reward_scales"]["similar_to_default"] = -0.1   # Stronger pose penalty - keeps natural posture
        reward_cfg["reward_scales"]["symmetry"] = -0.15            # Penalize asymmetric leg movement - reduces drift and wobble
        reward_cfg["reward_scales"]["lin_vel_z"] = -0.2            # Stronger bounce penalty

        env_cfg["episode_length_s"] = 20.0
        env_cfg["use_obstacles"] = False
        
        # Relaxed termination for learning to walk
        env_cfg["termination_if_pitch_greater_than"] = 40
        env_cfg["termination_if_roll_greater_than"] = 40
        env_cfg["termination_if_base_height_lower_than"] = 0.28  # Near-disabled: pitch/roll catch falls; height was terminating normal stride dips

    elif curriculum_stage == 3:
        # Stage 3: Goal-reaching with HIGH obstacle avoidance (use --dynamic_obstacles)
        command_cfg["lin_vel_x_range"] = [0.3, 0.8]
        command_cfg["lin_vel_y_range"] = [-0.3, 0.3]  # Allow lateral movement for avoidance
        command_cfg["ang_vel_range"] = [-0.5, 0.5]    # Allow turning
    
        reward_cfg["tracking_sigma"] = 0.25

        # Stability - LOW: just enough to prevent falling, not enough to reward-hack by standing
        reward_cfg["reward_scales"]["upright"] = 0.5               # Reduced: standing after avoidance must not be a local optimum
        reward_cfg["reward_scales"]["height"] = 0.2                # Reduced: same reason
        reward_cfg["reward_scales"]["symmetry"] = -0.01
        
        # Goal-reaching rewards - progress-driven, not proximity-hovering
        reward_cfg["reward_scales"]["target_proximity"] = 8.0      # Increased: linear gradient (1-dist/10) must overpower keep_moving at long range
        reward_cfg["reward_scales"]["target_reached"] = 20.0       # Strong success bonus
        reward_cfg["reward_scales"]["target_progress"] = 10.0       # Dominant - must MOVE toward goal, not just be near it
        reward_cfg["reward_scales"]["obstacle_avoidance"] = 2.0    # BALANCED - provides safety feedback without encouraging excessive retreat; lower than progress rewards to maintain forward drive
        
        # Movement quality
        reward_cfg["reward_scales"]["action_rate"] = -0.05         # Smooth movement
        reward_cfg["reward_scales"]["similar_to_default"] = -0.01  # Natural posture
        reward_cfg["reward_scales"]["lin_vel_z"] = -0.1            # Reduce bouncing
        
        # DISABLE competing rewards for goal-reaching
        reward_cfg["reward_scales"]["forward_movement"] = 0.0      # DISABLED - X-velocity check penalizes lateral avoidance maneuvers
        reward_cfg["reward_scales"]["keep_moving"] = 0.1           # Minimal: only breaks wall-lean local optimum; reduced so it doesn't outcompete proximity gradient
        reward_cfg["reward_scales"]["blocker_clearance"] = 3.0     # Rewards navigating past blocker X; fills reward-dead zone during lateral avoidance maneuver
        reward_cfg["reward_scales"]["goal_heading"] = 3.0          # Re-orient toward goal after passing blocker; fires at all ranges
        reward_cfg["reward_scales"]["tracking_lin_vel"] = 0.0
        reward_cfg["reward_scales"]["tracking_ang_vel"] = 0.0

        env_cfg["episode_length_s"] = 45.0
        env_cfg["use_obstacles"] = True
        env_cfg["terminate_on_collision"] = True  # STRICT: High obstacles must be avoided - episode failure teaches clear boundaries
        
        # Relaxed termination for learning navigation
        env_cfg["termination_if_pitch_greater_than"] = 30  # Allow more tilt while learning
        env_cfg["termination_if_roll_greater_than"] = 30
        env_cfg["termination_if_base_height_lower_than"] = 0.25  # Allow slight crouch
        
        # Dynamic obstacle configuration (HIGH obstacle - must avoid)
        env_cfg["dynamic_obstacle_height"] = 0.3   # Tall - forces avoidance (RED)
        env_cfg["dynamic_obstacle_width"] = 0.4
        env_cfg["dynamic_obstacle_distance"] = 4.0  # Closer: robot reaches it in ~8s at 0.5m/s (within 600-step episode)
    
    elif curriculum_stage == 4:
        # Stage 4: Goal-reaching with LOW obstacle jumping (use --dynamic_obstacles)
        command_cfg["lin_vel_x_range"] = [0.3, 0.6]  # Moderate speed
        command_cfg["lin_vel_y_range"] = [-0.1, 0.1]  # Minimal lateral
        command_cfg["ang_vel_range"] = [-0.2, 0.2]    # Minimal turning
    
        reward_cfg["tracking_sigma"] = 0.25
        reward_cfg["jump_height_threshold"] = 0.08  # Obstacles below this = low (jumpable)

        # Stability
        reward_cfg["reward_scales"]["upright"] = 3.0
        reward_cfg["reward_scales"]["height"] = 1.5
        
        # Goal-reaching rewards (active with --dynamic_obstacles)
        reward_cfg["reward_scales"]["target_proximity"] = 3.0      # Guide to target
        reward_cfg["reward_scales"]["target_reached"] = 15.0       # Success bonus
        
        # Jumping rewards - PRIMARY for low obstacles
        reward_cfg["reward_scales"]["jump_clearance"] = 10.0       # Reward jumping over
        reward_cfg["reward_scales"]["landing_stability"] = 3.0     # Stable landing
        
        # Movement quality
        reward_cfg["reward_scales"]["action_rate"] = -0.05
        reward_cfg["reward_scales"]["lin_vel_z"] = -0.1           # Allow vertical movement
        reward_cfg["reward_scales"]["similar_to_default"] = -0.01
        reward_cfg["reward_scales"]["symmetry"] = -0.05
        
        # DISABLE avoidance - we want robot to jump, not avoid
        reward_cfg["reward_scales"]["obstacle_avoidance"] = 0.0
        reward_cfg["reward_scales"]["forward_movement"] = 0.0
        reward_cfg["reward_scales"]["tracking_lin_vel"] = 0.0
        reward_cfg["reward_scales"]["tracking_ang_vel"] = 0.0

        env_cfg["episode_length_s"] = 30.0
        env_cfg["use_obstacles"] = True
        env_cfg["terminate_on_collision"] = False
        
        # Dynamic obstacle configuration (LOW obstacle - can jump)
        env_cfg["dynamic_obstacle_height"] = 0.06   # Low enough to jump (YELLOW)
        env_cfg["dynamic_obstacle_width"] = 1.0     # Wide - force jumping, not sidestepping
        env_cfg["dynamic_obstacle_distance"] = 3.0  # Slightly closer
        
        # Relaxed termination for jumping
        env_cfg["termination_if_pitch_greater_than"] = 25
        env_cfg["termination_if_roll_greater_than"] = 25
        env_cfg["termination_if_base_height_lower_than"] = 0.18

    elif curriculum_stage == 5:
        # Stage 5: Full navigation
        reward_cfg = {
            "jump_height_threshold": 0.08,
            "base_height_target": 0.37,
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
                
                # Base rewards (reduced for navigation)
                "lin_vel_z": 1.0,
                "action_rate": -0.1,
                "similar_to_default": -0.05,
            },
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

def ensure_exp_name_has_date(exp_name):
    # Regex: 8 digits at end of string
    if re.search(r'\d{8}$', exp_name):
        return exp_name
    else:
        return f"{exp_name}_{datetime.now().strftime('%Y%m%d')}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2-navigation")
    parser.add_argument("-B", "--num_envs", type=int, default=2048)
    parser.add_argument("--max_iterations", type=int, default=500)
    parser.add_argument("--uneven_terrain", action='store_true')
    parser.add_argument("--wind_force", action='store_true')
    parser.add_argument("--curriculum_stage", type=int, default=1)
    parser.add_argument("--resume", action='store_true')
    parser.add_argument("--resume_path", type=str, default=None)
    parser.add_argument("--params_pkl", type=str, default=None)
    parser.add_argument("--dynamic_obstacles", action='store_true')
    args = parser.parse_args()

    gs.init(logging_level="warning")
    
    exp_name = ensure_exp_name_has_date(args.exp_name)

    log_dir = f"logs/{exp_name}"
    if os.path.exists(log_dir) and not args.resume:
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    env_cfg, obs_cfg, reward_cfg, command_cfg = get_navigation_cfgs(curriculum_stage=args.curriculum_stage)
    train_cfg = get_navigation_train_cfg(args.exp_name, args.max_iterations)

    # Stage-specific entropy - standing needs low entropy to converge,
    # walking/navigation needs high entropy to explore new behaviors
    stage_entropy = {
        1: 0.001,  # Standing: very low - policy must converge and stay converged
        2: 0.02,   # Walking: high - must explore new gait behaviors from standing init
        3: 0.005,   # Navigation: moderate - some exploration for avoidance strategies
        4: 0.01,   # Jumping: moderate - some exploration for jump timing
        5: 0.005,  # Full navigation: low - refine learned behaviors
    }
    train_cfg["algorithm"]["entropy_coef"] = stage_entropy.get(args.curriculum_stage, 0.01)
    
    optimized_params_pkl = None
    if args.params_pkl is not None:
        with open(args.params_pkl, 'rb') as f:
            optimized_params_pkl = pickle.load(f)

            print("Applying optimized parameters from pickle file...")
            # Update train_cfg with optimized_params_pkl
            train_cfg["algorithm"]["learning_rate"] = optimized_params_pkl.get("learning_rate", train_cfg["algorithm"]["learning_rate"])
            train_cfg["algorithm"]["clip_param"] = optimized_params_pkl.get("clip_param", train_cfg["algorithm"]["clip_param"])
            train_cfg["algorithm"]["entropy_coef"] = optimized_params_pkl.get("entropy_coef", train_cfg["algorithm"]["entropy_coef"])
            train_cfg["algorithm"]["gamma"] = optimized_params_pkl.get("gamma", train_cfg["algorithm"]["gamma"])
            train_cfg["algorithm"]["value_loss_coef"] = optimized_params_pkl.get("value_loss_coef", train_cfg["algorithm"]["value_loss_coef"])
            train_cfg["algorithm"]["num_learning_epochs"] = optimized_params_pkl.get("num_learning_epochs", train_cfg["algorithm"]["num_learning_epochs"])
            
            # base rewards
            reward_cfg["reward_scales"]["lin_vel_z"] = optimized_params_pkl.get("lin_vel_z_scale", reward_cfg["reward_scales"]["lin_vel_z"])
            reward_cfg["reward_scales"]["action_rate"] = optimized_params_pkl.get("action_rate_scale", reward_cfg["reward_scales"]["action_rate"])
            reward_cfg["reward_scales"]["similar_to_default"] = optimized_params_pkl.get("similar_to_default_scale", reward_cfg["reward_scales"]["similar_to_default"])

            # navigation rewards
            reward_cfg["reward_scales"]["tracking_lin_vel"] = optimized_params_pkl.get("tracking_lin_vel_scale", reward_cfg["reward_scales"]["tracking_lin_vel"])
            reward_cfg["reward_scales"]["tracking_ang_vel"] = optimized_params_pkl.get("tracking_ang_vel_scale", reward_cfg["reward_scales"]["tracking_ang_vel"])
            reward_cfg["reward_scales"]["forward_movement"] = optimized_params_pkl.get("forward_movement_scale", reward_cfg["reward_scales"]["forward_movement"])
            reward_cfg["reward_scales"]["straight_walk_when_clear"] = optimized_params_pkl.get("straight_walk_when_clear_scale", reward_cfg["reward_scales"]["straight_walk_when_clear"])
            reward_cfg["reward_scales"]["adaptive_base_height"] = optimized_params_pkl.get("adaptive_base_height_scale", reward_cfg["reward_scales"]["adaptive_base_height"])
            reward_cfg["reward_scales"]["landing_stability"] = optimized_params_pkl.get("landing_stability_scale", reward_cfg["reward_scales"]["landing_stability"])
            reward_cfg["reward_scales"]["jumping_behavior"] = optimized_params_pkl.get("jumping_behavior_scale", reward_cfg["reward_scales"]["jumping_behavior"])
            reward_cfg["reward_scales"]["orientation_stability"] = optimized_params_pkl.get("orientation_stability_scale", reward_cfg["reward_scales"]["orientation_stability"])
            reward_cfg["reward_scales"]["angular_velocity_stability"] = optimized_params_pkl.get("angular_velocity_stability_scale", reward_cfg["reward_scales"]["angular_velocity_stability"])
            reward_cfg["reward_scales"]["upright_posture"] = optimized_params_pkl.get("upright_posture_scale", reward_cfg["reward_scales"]["upright_posture"])
            reward_cfg["reward_scales"]["ground_clearance"] = optimized_params_pkl.get("ground_clearance_scale", reward_cfg["reward_scales"]["ground_clearance"])
            reward_cfg["reward_scales"]["obstacle_avoidance"] = optimized_params_pkl.get("obstacle_avoidance_scale", reward_cfg["reward_scales"]["obstacle_avoidance"])
            
            # Update env_cfg if present
            env_cfg["obstacle_density"] = optimized_params_pkl.get("obstacle_density", env_cfg["obstacle_density"])
            env_cfg["episode_length_s"] = optimized_params_pkl.get("episode_length_s", env_cfg["episode_length_s"])

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
        wind_force=args.wind_force,
        dynamic_obstacles=args.dynamic_obstacles,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)

    if args.resume_path and os.path.exists(args.resume_path):
        print(f"🔄 Resuming training from: {args.resume_path}")
        runner.load(args.resume_path)

    elif args.resume:
        # Find latest checkpoint in current experiment
        model_files = [f for f in os.listdir(log_dir) if f.startswith('model_') and f.endswith('.pt')]
        if model_files:
            # Sort by checkpoint number
            model_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
            latest_model = model_files[-1]
            resume_path = os.path.join(log_dir, latest_model)
            print(f"🔄 Resuming from latest checkpoint: {resume_path}")
            runner.load(resume_path)

    with tqdm(total=args.max_iterations, desc="Training Progress") as pbar:
        # for iteration in range(args.max_iterations):
        is_resuming = args.resume or (args.resume_path is not None)
        runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True, pbar=pbar, resume_flag=is_resuming)
    # runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()

# Usage:
# python go2_train_navigation.py -e go2-nav --num_envs 2048 --max_iterations 500
# python go2_train_navigation.py -e go2-nav-terrain --uneven_terrain --wind_force