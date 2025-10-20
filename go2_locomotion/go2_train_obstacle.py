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
from importlib import metadata
import csv
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


def get_train_cfg(exp_name, max_iterations):
    train_cfg_dict = {
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
        },'obstacle_avoidance_scale': 0.504360534641974,
        "init_member_classes": {},
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "init_noise_std": 1.0,
            "class_name": "ActorCritic",
        },
        "runner": {
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "record_interval": 50,
            "resume": False,
            "resume_path": None,
            "run_name": "",
            "log_rewards": True,
        },
        "runner_class_name": "OnPolicyRunner",
        "num_steps_per_env": 24,
        "save_interval": 100,
        "empirical_normalization": None,
        "seed": 1,

        "reward_logging": {
            "log_individual_rewards": True,     # Log each reward component
            "reward_names": [                   # Specify which rewards to track
                "tracking_lin_vel",
                "tracking_ang_vel", 
                "lin_vel_z",
                "base_height",
                "action_rate",
                "similar_to_default",
                "obstacle_avoidance",
                "forward_progress"
            ]
        }
    }

    return train_cfg_dict


def get_cfgs():
    env_cfg = {
        "num_actions": 12,
        # joint/link names
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
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",
            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",
            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint",
        ],
        # PD
        "kp": 20.0,
        "kd": 0.5,
        # termination
        "termination_if_roll_greater_than": 10,  # degree
        "termination_if_pitch_greater_than": 10,
        # base pose
        "base_init_pos": [0.0, 0.0, 0.42],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 23.865854789774673,  # Longer episodes for obstacle navigation
        "resampling_time_s": 6.0,  # Longer command duration
        "action_scale": 0.25,
        "simulate_action_latency": True,
        "clip_actions": 1.0,

        'use_obstacles': True,
        'obstacle_density': 0.01581711235886657,  # Start with fewer obstacles for easier learning
        'obstacle_types': ['box', 'cylinder'],
        'obstacle_height_range': [0.05, 0.12],  # Slightly lower obstacles
        'obstacle_width_range': [0.1, 0.25],    # Slightly smaller obstacles
        'obstacle_spacing_min': 1.5,             # More spacing between obstacles
        'terrain_size': [25.0, 25.0],           # Larger terrain
        'clear_radius': 3.0,                     # Larger clear area around spawn
    }
    obs_cfg = {
        "num_obs": 45,
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
    }
    reward_cfg = {
        "tracking_sigma": 0.3,
        "base_height_target": 0.3,
        "feet_height_target": 0.075,
        "jump_height_threshold":0.1,
        "jump_reward_height":0.17,
        "reward_scales": {
            "tracking_lin_vel": 0.710311769413351,
            "tracking_ang_vel": 0.2,
            "lin_vel_z": -.5,
            "base_height": -10.0,
            "action_rate": 0.14112594271747939,
            "similar_to_default": -0.1,
            'obstacle_avoidance': 0.504360534641974,   # Reward for avoiding obstacles
            'forward_progress': 0.1,     # Reward for forward movement
            'landing_stability':0.2,
            'jump_timing':0.3,
            'jump_reward':0.5,
        },
    }
    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [0.3, 0.8],  # Variable forward speed for obstacle navigation
        "lin_vel_y_range": [-0.2, 0.2], # Allow lateral movement for obstacle avoidance
        "ang_vel_range": [-0.3, 0.3],   # Allow turning for obstacle navigation
    } # separate updates based on training needs
    env_cfg.update(get_randomization_cfg(strategy='delayed'))

    return env_cfg, obs_cfg, reward_cfg, command_cfg

def get_randomization_cfg(strategy=None):
    if strategy == 'delayed':
        return {
            'randomize_every_n_episodes': 2000,      # Randomize every 2000 episodes
            'interval_reset': 150,
            'seed_variation': True,
        }
    elif strategy == 'performance':
        return {
            'performance_threshold': 2.0,     # Performance threshold for performance-based randomization
            'min_random_gap': 100,             # Minimum gap between randomizations
        }
    elif strategy == 'curriculum':
        return {
            # Placeholder for curriculum-based parameters
        }
    else:
        return {}

def evaluate_during_training(runner, env_cfg, obs_cfg, reward_cfg, command_cfg, eval_episodes=5):
    """Quick evaluation during training"""
    
    # Create smaller evaluation environment with different obstacles
    eval_env_cfg = env_cfg.copy()
    eval_env_cfg.update({
        'obstacle_density': env_cfg['obstacle_density'] * 1.2,  # Slightly different
        'randomize_obstacles_per_episode': False,  # Fixed for consistent evaluation
    })
    
    eval_env = Go2Env(
        num_envs=32,  # Smaller batch for eval
        env_cfg=eval_env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=False  # Headless
    )

    if eval_env_cfg.get('randomize_obstacles_per_evaluation', False):
        eval_env.force_randomize_obstacles()
    
    policy = runner.get_inference_policy(device=gs.device)
    
    episode_rewards = []
    obs, _ = eval_env.reset()
    current_rewards = torch.zeros(eval_env.num_envs, device=eval_env.device)
    episodes_completed = 0
    
    with torch.no_grad():
        for step in range(500):  # Max 500 steps
            actions = policy(obs)
            obs, rewards, dones, infos = eval_env.step(actions)
            current_rewards += rewards
            
            if dones.any():
                for env_idx in torch.where(dones)[0]:
                    episode_rewards.append(current_rewards[env_idx].item())
                    current_rewards[env_idx] = 0
                    episodes_completed += 1
                    
                    if episodes_completed >= eval_episodes:
                        break
            
            if episodes_completed >= eval_episodes:
                break
    
    # Clean up evaluation environment
    del eval_env
    
    return {
        'mean_reward': np.mean(episode_rewards) if episode_rewards else 0,
        'std_reward': np.std(episode_rewards) if episode_rewards else 0,
        'episodes': len(episode_rewards)
    }

def delayed_randomization(env, episode_counter, randomize_interval):
    """Randomize obstacles if the episode counter exceeds the interval"""
    if episode_counter >= randomize_interval:
        env.force_randomize_obstacles()
        episode_counter = 0
    return episode_counter

def performance_randomization(env, iterations_done, performance_metric, threshold, last_randomization):
    """Randomize obstacles based on performance metric"""
    if performance_metric > threshold:
        env.force_randomize_obstacles()
        last_randomization = iterations_done
    return last_randomization

# not correctly implemented yet TODO
def get_randomization_schedule(iteration):
    """Get randomization interval based on training progress"""
    if iteration < 100:
        return float('inf')  # No randomization early
    elif iteration < 200:
        return 5000  # Very infrequent
    elif iteration < 350:
        return 2000  # Moderate
    else:
        return 1000  # More frequent for final training

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2-obstacles")
    parser.add_argument("-B", "--num_envs", type=int, default=2048)  # Fewer envs to start
    parser.add_argument("--max_iterations", type=int, default=500)   # More iterations for obstacle learning
    parser.add_argument("--eval_interval", type=int, default=50)     # Evaluate every 50 iterations
    args = parser.parse_args()

    gs.init(logging_level="warning")

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    env = Go2Env(
        num_envs=args.num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, command_cfg=command_cfg
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)

    # remove from here on out if logging does not work

    # Set up evaluation log file
    eval_log_file = f"{log_dir}/{args.exp_name}_training_evaluation.csv"
    with open(eval_log_file, 'w', newline='') as csvfile:
        fieldnames = ['iteration', 'training_reward', 'eval_mean_reward', 'eval_std_reward', 'overfitting_gap']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    
    iterations_done = 0

    randomize_interval = env_cfg.get('randomize_every_n_episodes', 1500)

    randomize_strategy = env_cfg.get('randomize_strategy', 'delayed')

    if randomize_strategy == 'delayed':
        episode_counter = 0
        reset_iteration = env_cfg.get('interval_reset', 150)
    elif randomize_strategy == 'performance':
        last_randomization = 0
        min_random_gap = env_cfg.get('min_random_gap', 100)
        performance_threshold = env_cfg.get('performance_threshold', 2.0)
    elif randomize_strategy == 'curriculum': # Placeholder for curriculum-based strategy TODO
        pass 

    while iterations_done < args.max_iterations:
        current_batch = min(args.eval_interval, args.max_iterations - iterations_done)
        runner.learn(num_learning_iterations=current_batch, init_at_random_ep_len=True)
        iterations_done += current_batch

        if randomize_strategy == 'delayed':
            if iterations_done > reset_iteration:
                episode_counter += args.num_envs * current_batch
                episode_counter = delayed_randomization(env, episode_counter, randomize_interval)
        elif randomize_strategy == 'performance':
            if (iterations_done - last_randomization) >= min_random_gap:
                try:
                    recent_reward = env.rew_buf.mean().item()
                    last_randomization = performance_randomization(env, iterations_done, recent_reward, performance_threshold, last_randomization)
                except:
                    pass
        elif randomize_strategy == 'curriculum':
            reset_iteration = 0 # Placeholder for curriculum-based strategy TODO


        """
        episode_counter += args.num_envs * current_batch

        if episode_counter >= randomize_interval:
            env.force_randomize_obstacles()
            episode_counter = 0
        """

        # Get current training performance; evaluation is failing, idk why
        try:
            # Method 1: Get from runner's storage (most reliable for PPO)
            if hasattr(runner, 'alg') and hasattr(runner.alg, 'storage'):
                # Get recent rewards from the rollout buffer
                rewards = runner.alg.storage.rewards  # Shape: [num_steps, num_envs]
                training_reward = rewards.mean().item()
                print(f"✓ Method 1: Got training reward from runner storage: {training_reward:.3f}")
            else:
                raise AttributeError("No runner storage available")
        except Exception as e:
            print(f"Method 1 failed: {e}")
            try:
                # Method 2: Get from environment's current reward buffer
                training_reward = env.rew_buf.mean().item()
                print(f"✓ Method 2: Got training reward from env.rew_buf: {training_reward:.3f}")
            except Exception as e:
                print(f"Method 2 failed: {e}")
                try:
                    # Method 3: Get from environment's episode sums
                    total_reward = sum(env.episode_sums[key].mean().item() for key in env.episode_sums.keys())
                    training_reward = total_reward
                    print(f"✓ Method 3: Got training reward from env.episode_sums: {training_reward:.3f}")
                except Exception as e:
                    print(f"Method 3 failed: {e}")
                    # Method 4: Ultimate fallback
                    print("⚠️  All methods failed, using 0 for training reward")
                    training_reward = 0.0
        
        # Quick evaluation
        print(f"🔍 Evaluating at iteration {iterations_done}...")
        eval_stats = evaluate_during_training(runner, env_cfg, obs_cfg, reward_cfg, command_cfg)
        
        # Calculate overfitting gap
        overfitting_gap = training_reward - eval_stats['mean_reward']
        
        # Log results
        with open(eval_log_file, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['iteration', 'training_reward', 'eval_mean_reward', 'eval_std_reward', 'overfitting_gap'])
            writer.writerow({
                'iteration': iterations_done,
                'training_reward': training_reward,
                'eval_mean_reward': eval_stats['mean_reward'],
                'eval_std_reward': eval_stats['std_reward'],
                'overfitting_gap': overfitting_gap
            })
    # runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()

"""
# training with obstacles
python examples/locomotion/go2_train_obstacle.py -e go2-obstacles --num_envs 2048 --max_iterations 500

# training with fewer environments for testing
python examples/locomotion/go2_train_obstacle.py -e go2-obstacles --num_envs 512 --max_iterations 100
"""
