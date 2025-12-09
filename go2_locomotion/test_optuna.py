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
# from go2_env import Go2Env
from go2_env_navigation import Go2NavigationEnv
from go2_env_petting import Go2PettingEnv
from rsl_rl.runners import OnPolicyRunner
import torch
from go2_train_navigation import get_navigation_cfgs
from go2_train_petting import get_petting_cfgs
import traceback
import argparse
from datetime import datetime
from upload_file_to_gdrive import authenticate_google_drive, upload_file, send_discord_notification, zip_log_folder
from dotenv import load_dotenv

load_dotenv()

import sys
# from go2_train_obstacle import main as train_main

from dotenv import load_dotenv
load_dotenv()

webhook_url = os.getenv("DISCORD_WEBHOOK_URL")

def start_training_with_params(filename, study_name):
    # Modify sys.argv to pass arguments to the training script
    original_argv = sys.argv.copy()
    sys.argv = [
        'go2_train_obstacle.py',
        '-e', f'{study_name}_optimized',
        '--num_envs', '3072',
        '--max_iterations', '2000',
        '--params_pkl', filename
    ]
    
    try:
        # Call the training main function directly
        # train_main()
        print("✅ Training completed successfully!")

        log_folder = f"logs/{study_name}_optimized"
        if os.path.exists(log_folder):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            zip_filename = f"logs/{study_name}_optimized_{timestamp}.zip"
            
            if zip_log_folder(log_folder, zip_filename):
                # Upload zipped logs to Google Drive
                try:
                    service = authenticate_google_drive()
                    result = upload_file(service, zip_filename, 
                                       drive_filename=os.path.basename(zip_filename))
                    if result:
                        send_discord_notification(webhook_url, f"✅ Training logs uploaded to Google Drive: {result.get('webViewLink')}")
                        return result.get('webViewLink')
                    else:
                        send_discord_notification(webhook_url, "❌ Failed to upload training logs")
                except Exception as e:
                    send_discord_notification(webhook_url, f"❌ Error uploading training logs: {e}")
            else:
                send_discord_notification(webhook_url, f"❌ Failed to zip log folder: {log_folder}")
        else:
            send_discord_notification(webhook_url,f"❌ Log folder not found: {log_folder}")
    except Exception as e:
        send_discord_notification(webhook_url, f"❌ Training failed: {e}")
    finally:
        # Restore original argv
        sys.argv = original_argv

def objective(trial, num_envs=256, training_env = 'navigation'):
    """Optuna objective function for hyperparameter optimization"""
    
    env = None
    runner = None
    log_dir = None

    # Sample hyperparameters
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 5e-4, log=True)
    clip_param = trial.suggest_float('clip_param', 0.15, 0.3)
    entropy_coef = trial.suggest_float('entropy_coef', 0.005, 0.02, log=True)
    gamma = trial.suggest_float('gamma', 0.98, 0.998)
    value_loss_coef = trial.suggest_float('value_loss_coef', 0.5, 1.5)
    num_learning_epochs = trial.suggest_int('num_learning_epochs', 3, 10)
    
    # base environment reward scales
    lin_vel_z_scale = trial.suggest_float('lin_vel_z_scale', 0.5, 2.5)
    action_rate_scale = trial.suggest_float('action_rate_scale', -0.02, -0.001)
    similar_to_default_scale = trial.suggest_float('similar_to_default_scale', 0.1, 1.0)

    if training_env == 'navigation':
        # navigation environment reward scales
        tracking_lin_vel_scale = trial.suggest_float('tracking_lin_vel_scale', 1.0, 3.0)
        tracking_ang_vel_scale = trial.suggest_float('tracking_ang_vel_scale', 0.5, 2.0)
        forward_movement_scale = trial.suggest_float('forward_movement_scale', 0.1, 2.0)
        straight_walk_when_clear_scale = trial.suggest_float('straight_walk_when_clear_scale', 0.1, 2.0)
        adaptive_base_height_scale = trial.suggest_float('adaptive_base_height_scale', 0.1, 2.0)
        landing_stability_scale = trial.suggest_float('landing_stability_scale', 0.1, 2.0)
        jumping_behavior_scale = trial.suggest_float('jumping_behavior_scale', 0.1, 2.0)
        orientation_stability_scale = trial.suggest_float('orientation_stability_scale', 0.1, 2.0)
        angular_velocity_stability_scale = trial.suggest_float('angular_velocity_stability_scale', 0.1, 2.0)
        upright_posture_scale = trial.suggest_float('upright_posture_scale', 0.1, 2.0)
        ground_clearance_scale = trial.suggest_float('ground_clearance_scale', 0.1, 2.0)
        obstacle_avoidance_scale = trial.suggest_float('obstacle_avoidance_scale', 0.1, 1.0)

    else: # pet robot
        petting_response_scale = trial.suggest_float('petting_response_scale', 0.1, 1.0)
        petting_stability_scale = trial.suggest_float('petting_stability_scale', .05, .5)
        calm_behavior_scale = trial.suggest_float('calm_behavior_scale', 0.02, .30)
        flexible_height_scale = trial.suggest_float('flexible_height_scale', 0.02, .30)

        petting_probability = trial.suggest_float('petting_probability', 0.005, 0.02)        # 0.5% - 2%
        petting_duration = trial.suggest_int('petting_duration', 30, 100)                    # 0.6-2 seconds at 50Hz

    if training_env == 'petting':
        env_cfg, obs_cfg, reward_cfg, command_cfg = get_petting_cfgs()

        # Environment parameters
        episode_length = trial.suggest_float('episode_length_s', 15.0, 20.0)

        reward_cfg['reward_scales']['petting_response'] = petting_response_scale
        reward_cfg['reward_scales']['petting_stability'] = petting_stability_scale
        reward_cfg['reward_scales']['calm_behavior'] = calm_behavior_scale
        reward_cfg['reward_scales']['flexible_height'] = flexible_height_scale

        env_cfg['petting_probability'] = petting_probability
        env_cfg['petting_duration'] = petting_duration
    else:
        env_cfg, obs_cfg, reward_cfg, command_cfg = get_navigation_cfgs()
    
        # Environment parameters
        obstacle_density = trial.suggest_float('obstacle_density', 0.01, 0.05)
        episode_length = trial.suggest_float('episode_length_s', 15.0, 40.0)

        env_cfg['obstacle_density'] = obstacle_density

        reward_cfg['reward_scales']['tracking_lin_vel'] = tracking_lin_vel_scale
        reward_cfg['reward_scales']['tracking_ang_vel'] = tracking_ang_vel_scale
        reward_cfg['reward_scales']['forward_movement'] = forward_movement_scale
        reward_cfg['reward_scales']['straight_walk_when_clear'] = straight_walk_when_clear_scale
        reward_cfg['reward_scales']['adaptive_base_height'] = adaptive_base_height_scale
        reward_cfg['reward_scales']['landing_stability'] = landing_stability_scale
        reward_cfg['reward_scales']['jumping_behavior'] = jumping_behavior_scale
        reward_cfg['reward_scales']['orientation_stability'] = orientation_stability_scale
        reward_cfg['reward_scales']['angular_velocity_stability'] = angular_velocity_stability_scale
        reward_cfg['reward_scales']['upright_posture'] = upright_posture_scale
        reward_cfg['reward_scales']['ground_clearance'] = ground_clearance_scale
        reward_cfg['reward_scales']['obstacle_avoidance'] = obstacle_avoidance_scale
    

    env_cfg['episode_length_s'] = episode_length

    reward_cfg['reward_scales']['lin_vel_z'] = lin_vel_z_scale
    reward_cfg['reward_scales']['action_rate'] = action_rate_scale
    reward_cfg['reward_scales']['similar_to_default'] = similar_to_default_scale
    
    # Create unique experiment name
    exp_name = f"optuna_trial_{trial.number}"
    log_dir = f"logs/hyperopt/{exp_name}"
    
    try:

        # EXPLICIT CLEANUP
        
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        if training_env == 'navigation':
        # Initialize environment and train
            env = Go2NavigationEnv(num_envs=num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg, 
                        reward_cfg=reward_cfg, command_cfg=command_cfg, wind_force=False,
                        uneven_terrain=False)
        else:
            env = Go2PettingEnv(num_envs=num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg, 
                        reward_cfg=reward_cfg, command_cfg=command_cfg)

        train_cfg = get_train_cfg_optimized(trial, learning_rate, clip_param, 
                                        entropy_coef, gamma, value_loss_coef, 
                                        num_learning_epochs, training_env)

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
    
    finally:

        try:
            if env is not None:
                if hasattr(env, 'scene'):
                    env.scene.reset()  # Reset scene state
                if hasattr(env, 'close'):
                    env.close()
                del env
            if runner is not None:
                if hasattr(runner, 'alg') and hasattr(runner.alg, 'actor_critic'):
                    del runner.alg.actor_critic
                if hasattr(runner, 'alg'):
                    del runner.alg
                del runner
            
            import gc
            gc.collect()

            # Clear GPU memory
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Clean up log directory
            if os.path.exists(log_dir):
                shutil.rmtree(log_dir)
                
        except Exception as cleanup_error:
            print(f"Cleanup error in trial {trial.number}: {cleanup_error}")
            torch.cuda.empty_cache()

def get_train_cfg_optimized(trial, lr, clip_param, entropy_coef, gamma, 
                           value_loss_coef, num_learning_epochs, training_env='navigation'):
    
    num_steps = trial.suggest_int('num_steps_per_env', 16, 32, step=4)
    
    if training_env == 'petting':
        max_iterations = 200
    else:
        max_iterations = 100

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
            "max_iterations": max_iterations,
        },
        "num_steps_per_env": num_steps,
        "save_interval": 50,
        "seed": 1,
        "empirical_normalization": True, 
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--study_name", type=str, default="go2-obstacles")
    parser.add_argument("-T", "--n_trials", type=int, default=50)
    parser.add_argument("-N", "--num_envs", type=int, default=128)
    parser.add_argument("--training_env", type=str, choices=['navigation', 'petting'], default='navigation')
    args = parser.parse_args()

    gs.init(logging_level="warning")
    
    # Create study
    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=20))
    study.optimize(lambda trial: objective(trial, num_envs= args.num_envs, training_env=args.training_env), n_trials=args.n_trials)
    
    print("Best parameters:", study.best_params)
    print("Best value:", study.best_value)
    
    filename = f'logs/hyperopt/{args.study_name}_{datetime.now().strftime("%Y%m%d")}.pkl'
    # Save best parameters
    with open(filename, 'wb') as f:
        pickle.dump(study.best_params, f)

    try:
        message = (f"✅ Optuna study '{filename}' completed!\n"
                    "Now starting training with best hyperparameters.")
        send_discord_notification(webhook_url, message)
    except Exception as e:
        print(f"❌ Error sending Discord notification: {e}")
    
    # try:
    #     start_training_with_params(filename, args.study_name)
    #     message = (f"✅ Training with best hyperparameters from '{filename}' completed successfully!\n"
    #                f"Uploaded logs ({args.study_name}_optimized) to Google Drive.")

    # except Exception as e:
    #     message = (f"❌ Training with best hyperparameters from '{filename}' failed: {e}")

    # send_discord_notification(webhook_url, message)