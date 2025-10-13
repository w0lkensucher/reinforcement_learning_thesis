import genesis as gs
from go2_env import Go2Env
from rsl_rl.runners import OnPolicyRunner
import numpy as np
import pickle

class CurriculumTrainer:
    def __init__(self, base_cfg):
        self.base_cfg = base_cfg
        self.stages = self.define_curriculum_stages()
        self.current_stage = 0
        
    def define_curriculum_stages(self):
        """Define progressive training stages"""
        return [
            {  # Stage 1: Basic locomotion
                'name': 'basic_locomotion',
                'env_cfg': {
                    'use_obstacles': False,
                    'episode_length_s': 20.0,
                },
                'reward_cfg': {
                    'reward_scales': {
                        'tracking_lin_vel': 1.5,
                        'action_rate': -0.01,
                    }
                },
                'iterations': 200,
                'success_threshold': 0.8
            },
            {  # Stage 2: Simple obstacles
                'name': 'simple_obstacles',
                'env_cfg': {
                    'use_obstacles': True,
                    'obstacle_density': 0.015,
                    'obstacle_height_range': [0.03, 0.07],
                    'episode_length_s': 25.0,
                },
                'reward_cfg': {
                    'reward_scales': {
                        'tracking_lin_vel': 1.2,
                        'obstacle_avoidance': 0.5,
                        'action_rate': -0.005,
                    }
                },
                'iterations': 300,
                'success_threshold': 0.7
            },
            {  # Stage 3: Complex obstacles
                'name': 'complex_obstacles',
                'env_cfg': {
                    'use_obstacles': True,
                    'obstacle_density': 0.03,
                    'obstacle_height_range': [0.05, 0.12],
                    'episode_length_s': 30.0,
                },
                'reward_cfg': {
                    'reward_scales': {
                        'tracking_lin_vel': 1.0,
                        'obstacle_avoidance': 1.0,
                        'forward_progress': 0.3,
                        'action_rate': -0.005,
                    }
                },
                'iterations': 400,
                'success_threshold': 0.6
            }
        ]
    
    def train_stage(self, stage_idx, previous_checkpoint=None):
        """Train a specific curriculum stage"""
        stage = self.stages[stage_idx]
        print(f"Training stage {stage_idx + 1}: {stage['name']}")
        
        # Merge configurations
        env_cfg = {**self.base_cfg['env_cfg'], **stage['env_cfg']}
        reward_cfg = {**self.base_cfg['reward_cfg']}
        
        # Update reward scales
        for key, value in stage['reward_cfg']['reward_scales'].items():
            reward_cfg['reward_scales'][key] = value
        
        # Create training config
        train_cfg = self.base_cfg['train_cfg'].copy()
        train_cfg['runner']['max_iterations'] = stage['iterations']
        
        if previous_checkpoint:
            train_cfg['runner']['resume'] = True
            train_cfg['runner']['resume_path'] = previous_checkpoint
        
        # Setup logging
        exp_name = f"curriculum_{stage['name']}"
        log_dir = f"logs/curriculum/{exp_name}"
        
        # Train
        env = Go2Env(num_envs=2048, env_cfg=env_cfg, 
                    obs_cfg=self.base_cfg['obs_cfg'],
                    reward_cfg=reward_cfg,
                    command_cfg=self.base_cfg['command_cfg'])
        
        runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
        runner.learn(num_learning_iterations=stage['iterations'])
        
        # Evaluate success
        success_rate = self.evaluate_stage_success(runner, stage)
        
        checkpoint_path = f"{log_dir}/model.pt"
        return success_rate >= stage['success_threshold'], checkpoint_path
    
    def evaluate_stage_success(self, runner, stage):
        """Evaluate if stage training was successful"""
        # Simple metric: average reward over last 50 iterations
        recent_rewards = runner.tot_timesteps[-50:]
        success_rate = np.mean(recent_rewards) / runner.max_episode_length
        return success_rate
    
    def train_curriculum(self):
        """Train through all curriculum stages"""
        checkpoint = None
        
        for i, stage in enumerate(self.stages):
            success, new_checkpoint = self.train_stage(i, checkpoint)
            
            if success:
                print(f"Stage {i+1} completed successfully!")
                checkpoint = new_checkpoint
            else:
                print(f"Stage {i+1} failed, retraining...")
                # Optionally retry or adjust parameters
                
        return checkpoint

# Usage
if __name__ == "__main__":
    from go2_train_obstacle import get_cfgs, get_train_cfg
    
    gs.init(logging_level="warning")
    
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg("curriculum", 500)
    
    base_cfg = {
        'env_cfg': env_cfg,
        'obs_cfg': obs_cfg,
        'reward_cfg': reward_cfg,
        'command_cfg': command_cfg,
        'train_cfg': train_cfg
    }
    
    trainer = CurriculumTrainer(base_cfg)
    final_checkpoint = trainer.train_curriculum()