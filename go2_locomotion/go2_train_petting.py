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

from rsl_rl.runners import OnPolicyRunner
import genesis as gs
from go2_env_petting import Go2PettingEnv


def get_petting_train_cfg(exp_name, max_iterations):
    return {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.1,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 3e-4,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 10,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [256, 128, 64],  # Smaller network for petting
            "critic_hidden_dims": [256, 128, 64],
            "init_noise_std": 0.8,
            "class_name": "ActorCritic",
        },
        "runner": {
            "experiment_name": exp_name,
            "max_iterations": max_iterations,
            "log_interval": 1,
            "record_interval": 25,
            "resume": False,
        },
        "num_steps_per_env": 32,  # Longer episodes for petting
        "save_interval": 50,
        "seed": 1,
    }


def get_petting_cfgs():
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
        "kp": 20.0,
        "kd": 0.5,
        "termination_if_roll_greater_than": 45,
        "termination_if_pitch_greater_than": 45,
        "base_init_pos": [0.0, 0.0, 0.42],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 30.0,  # Longer episodes for petting
        "resampling_time_s": 10.0,  # Longer command duration
        "action_scale": 0.25,
        "simulate_action_latency": True,
        "clip_actions": 1.0,
        
        # PETTING-SPECIFIC CONFIG
        'head_touch_radius': 0.2,
        'gentle_speed_threshold': 0.3,
        'head_height_offset': 0.25,
        'use_height_detection': True,
        'gentle_press_range': [0.02, 0.08],
        'gentle_vel_threshold': 0.1,
        'gesture_duration': 100,
        'cooldown_duration': 250,
    }
    
    obs_cfg = {
        "num_obs": 47,  # Base (45) + gesture_active (1) + touch_detected (1)
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
    }
    
    reward_cfg = {
        "base_height_target": 0.35,  # Slightly lower for petting
        "reward_scales": {
            # Petting-focused rewards
            "petting_response": 3.0,        # High reward for gestures
            "petting_stability": 2.0,       # Stability during gestures
            "calm_behavior": 1.0,           # Calm when not petted
            "flexible_height": 1.0,         # Allow height variation
            
            # Very reduced base rewards
            "lin_vel_z": 0.3,              # Allow some jumping for gestures
            "action_rate": 0.1,            # Allow expressive movements
            "similar_to_default": 0.05,    # Allow gesture poses
        },
    }
    
    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [-0.3, 0.5],  # Slower, gentle movement
        "lin_vel_y_range": [-0.2, 0.2],  # Minimal lateral
        "ang_vel_range": [-0.3, 0.3],    # Gentle turning
    }
    
    return env_cfg, obs_cfg, reward_cfg, command_cfg


def evaluate_petting_behavior(runner, env_cfg, obs_cfg, reward_cfg, command_cfg):
    """Evaluate petting response behavior"""
    eval_env = Go2PettingEnv(
        num_envs=32,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=False
    )
    
    policy = runner.get_inference_policy(device=gs.device)
    
    gesture_counts = []
    petting_responses = []
    
    obs, _ = eval_env.reset()
    
    with torch.no_grad():
        for step in range(1000):
            actions = policy(obs)
            obs, rewards, dones, infos = eval_env.step(actions)
            
            if hasattr(eval_env, 'gesture_timer'):
                active_gestures = (eval_env.gesture_timer > 0).sum().item()
                gesture_counts.append(active_gestures)
            
            if hasattr(eval_env, 'head_touched'):
                petting_detections = eval_env.head_touched.sum().item()
                petting_responses.append(petting_detections)
    
    del eval_env
    
    return {
        'avg_gestures_per_step': np.mean(gesture_counts) if gesture_counts else 0,
        'avg_petting_detections': np.mean(petting_responses) if petting_responses else 0,
        'total_gesture_activations': sum(gesture_counts),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2-petting")
    parser.add_argument("-B", "--num_envs", type=int, default=1024)  # Fewer envs for petting
    parser.add_argument("--max_iterations", type=int, default=300)
    parser.add_argument("--eval_petting", action='store_true')
    args = parser.parse_args()

    gs.init(logging_level="warning")
    
    log_dir = f"logs/{args.exp_name}"
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    env_cfg, obs_cfg, reward_cfg, command_cfg = get_petting_cfgs()
    train_cfg = get_petting_train_cfg(args.exp_name, args.max_iterations)

    # Save configs
    pickle.dump([env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
                open(f"{log_dir}/cfgs.pkl", "wb"))

    # Create PETTING environment
    env = Go2PettingEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    
    if args.eval_petting:
        # Train with periodic petting evaluation
        eval_interval = 25
        for i in range(0, args.max_iterations, eval_interval):
            batch_size = min(eval_interval, args.max_iterations - i)
            runner.learn(num_learning_iterations=batch_size, init_at_random_ep_len=True)
            
            # Evaluate petting behavior
            petting_stats = evaluate_petting_behavior(runner, env_cfg, obs_cfg, reward_cfg, command_cfg)
            print(f"Iteration {i+batch_size}: Avg gestures/step: {petting_stats['avg_gestures_per_step']:.3f}")
    else:
        runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()

# Usage:
# python go2_train_petting.py -e go2-pet --num_envs 1024 --max_iterations 300
# python go2_train_petting.py -e go2-pet-eval --eval_petting