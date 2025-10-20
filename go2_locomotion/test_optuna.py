import os
if os.name != 'nt':
    os.environ['SETUPTOOLS_USE_DISTUTILS'] = 'local'

    import sys
    if 'distutils' in sys.modules:
        del sys.modules['distutils']

    import importlib
    distutils_core = importlib.import_module('distutils.core')
    print("Distutils core loaded from:", distutils_core.__file__)

import optuna
import pickle
import shutil
import genesis as gs
from go2_env import Go2Env
from rsl_rl.runners import OnPolicyRunner
import torch
from go2_train_obstacle import get_cfgs
import traceback
import argparse
from datetime import datetime

def objective(trial):
    """Optuna objective function for hyperparameter optimization"""
    
    # Sample hyperparameters
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    clip_param = trial.suggest_float('clip_param', 0.1, 0.4)
    entropy_coef = trial.suggest_float('entropy_coef', 0.001, 0.1, log=True)
    gamma = trial.suggest_float('gamma', 0.98, 0.999)
    value_loss_coef = trial.suggest_float('value_loss_coef', 0.5, 3.0)
    num_learning_epochs = trial.suggest_int('num_learning_epochs', 3, 10)
    
    # Reward scale parameters
    tracking_lin_vel_scale = trial.suggest_float('tracking_lin_vel_scale', 0.5, 2.0)
    obstacle_avoidance_scale = trial.suggest_float('obstacle_avoidance_scale', 0.1, 1.0)
    action_rate_scale = trial.suggest_float('action_rate_scale', -0.02, -0.001)
    
    # Environment parameters
    obstacle_density = trial.suggest_float('obstacle_density', 0.01, 0.05)
    episode_length = trial.suggest_float('episode_length_s', 15.0, 40.0)
    
    # Create configurations with sampled parameters
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    
    # Update with trial parameters
    env_cfg['obstacle_density'] = obstacle_density
    env_cfg['episode_length_s'] = episode_length
    
    reward_cfg['reward_scales']['tracking_lin_vel'] = tracking_lin_vel_scale
    reward_cfg['reward_scales']['obstacle_avoidance'] = obstacle_avoidance_scale
    reward_cfg['reward_scales']['action_rate'] = action_rate_scale
    
    train_cfg = get_train_cfg_optimized(trial, learning_rate, clip_param, 
                                       entropy_coef, gamma, value_loss_coef, 
                                       num_learning_epochs)
    
    # Create unique experiment name
    exp_name = f"optuna_trial_{trial.number}"
    log_dir = f"logs/hyperopt/{exp_name}"
    
    try:
        # Initialize environment and train
        env = Go2Env(num_envs=512, env_cfg=env_cfg, obs_cfg=obs_cfg, 
                    reward_cfg=reward_cfg, command_cfg=command_cfg)
        
        runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
        runner.learn(num_learning_iterations=100, init_at_random_ep_len=True)
        
        # Get final reward as optimization target
        final_reward = torch.mean(env.episode_sums['reward']).item() 
        
        # Clean up
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
            
        return final_reward
        
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        traceback.print_exc()  # <-- This prints the full error traceback to the terminal
        return -1000  # Large penalty for failed trials

def get_train_cfg_optimized(trial, lr, clip_param, entropy_coef, gamma, 
                           value_loss_coef, num_learning_epochs):
    return {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": clip_param,
            "desired_kl": 0.01,
            "entropy_coef": entropy_coef,
            "gamma": gamma,
            "lam": 0.95,
            "learning_rate": lr,
            "max_grad_norm": 1.0,
            "num_learning_epochs": num_learning_epochs,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": value_loss_coef,
        },
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "init_noise_std": 1.0,
            "class_name": "ActorCritic",
        },
        "runner": {
            "experiment_name": f"optuna_trial_{trial.number}",
            "max_iterations": 100,
        },
        "num_steps_per_env": 24,
        "save_interval": 50,
        "seed": 1,
        "empirical_normalization": True, 
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--study_name", type=str, default="go2-obstacles")
    parser.add_argument("-B", "--n_trials", type=int, default=50)
    args = parser.parse_args()

    gs.init(logging_level="warning")
    
    # Create study
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=args.n_trials)
    
    print("Best parameters:", study.best_params)
    print("Best value:", study.best_value)
    
    # Save best parameters
    with open(f'logs/hyperopt/{args.study_name}_{datetime.now().strftime("%Y%m%d")}.pkl', 'wb') as f:
        pickle.dump(study.best_params, f)