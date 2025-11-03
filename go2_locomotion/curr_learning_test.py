import os
import genesis as gs
from go2_env import Go2Env
from rsl_rl.runners import OnPolicyRunner
import numpy as np
import pickle
import os
from datetime import datetime

class CurriculumTrainer:
    def __init__(self, base_cfg, exp_name="curriculum_training"):
        self.base_cfg = base_cfg
        self.exp_name = exp_name
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
            {  # Stage 2: Wind resistance
            'name': 'wind_resistance',
                'env_cfg': {
                    'use_obstacles': False,
                    'wind_force_range': [0.0, 5.0],
                    'episode_length_s': 20.0,
                },
                'reward_cfg': {
                    'reward_scales': {
                        'tracking_lin_vel': 1.3,
                        'wind_resistance': 0.2,
                        'action_rate': -0.01,
                    }
                },
                'iterations': 250,
                'success_threshold': 0.75
            },
            {   # Stage 3: Uneven terrain
                'name': 'uneven_terrain',
                'env_cfg': {
                    'use_obstacles': False,
                    'uneven_terrain': True,
                    'terrain_roughness': 0.1,
                    'episode_length_s': 22.0,
                },
                'reward_cfg': {
                    'reward_scales': {
                        'tracking_lin_vel': 1.3,
                        'uneven_terrain': 0.2,
                        'action_rate': -0.01,
                    }
                },
                'iterations': 250,
                'success_threshold': 0.75
            },
            {  # Stage 4: Simple obstacles
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
            {  # Stage 5: Complex obstacles
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
            }, 
            {   # Stage 6: jumping obstacles
                'name': 'jumping_obstacles',
                'env_cfg': {
                    'use_obstacles': True,
                    'obstacle_density': 0.04,
                    'obstacle_height_range': [0.1, 0.2],
                    'jumping_obstacles': True,
                    'episode_length_s': 35.0,
                },
                'reward_cfg': {
                    'reward_scales': {
                        'tracking_lin_vel': 0.8,
                        'obstacle_avoidance': 1.5,
                        'forward_progress': 0.4,
                        'action_rate': -0.002,
                    }
                },
                'iterations': 500,
                'success_threshold': 0.5
            }
        ]
    
    def train_stage(self, stage_idx, previous_checkpoint=None):
        """Train a specific curriculum stage"""
        stage = self.stages[stage_idx]
        print(f"\n{'='*50}")
        print(f"Training stage {stage_idx + 1}: {stage['name']}")
        print(f"📝 {stage['description']}")
        print(f"🔄 Training for {stage['iterations']} iterations")
        print(f"{'='*50}")

        import copy
        env_cfg = copy.deepcopy(self.base_cfg['env_cfg'])
        reward_cfg = copy.deepcopy(self.base_cfg['reward_cfg'])
        train_cfg = copy.deepcopy(self.base_cfg['train_cfg'])
        
        # Apply stage modifications
        for key, value in stage['env_cfg'].items():
            env_cfg[key] = value

        # Update reward scales
        for key, value in stage['reward_cfg']['reward_scales'].items():
            reward_cfg['reward_scales'][key] = value
        
        # Set training parameters
        train_cfg['runner']['max_iterations'] = stage['iterations']
        train_cfg['runner']['save_interval'] = min(50, stage['iterations'] // 4)
        
        # Setup logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = f"logs/{self.exp_name}_curriculum/stage_{stage_idx+1}_{stage['name']}_{timestamp}"
        os.makedirs(log_dir, exist_ok=True)
        
        try:
            # Create environment
            env = Go2Env(
                num_envs=train_cfg['runner'].get('num_envs', 2048),
                env_cfg=env_cfg,
                obs_cfg=self.base_cfg['obs_cfg'],
                reward_cfg=reward_cfg,
                command_cfg=self.base_cfg['command_cfg'],
                headless=True
            )
            
            # Create runner
            runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
                        
            # Load previous checkpoint if available
            if previous_checkpoint and os.path.exists(previous_checkpoint):
                print(f"📂 Loading checkpoint: {previous_checkpoint}")
                runner.load(previous_checkpoint)
            
            # Train
            print(f"🏃 Starting training...")
            runner.learn(num_learning_iterations=stage['iterations'])

            # Save checkpoint
            checkpoint_path = os.path.join(log_dir, "model.pt")
            runner.save(checkpoint_path)
            
            # Evaluate success
            success_rate = self.evaluate_stage_success(runner, stage)
            success = success_rate >= stage['success_threshold']
            
            print(f"\n📊 Stage {stage_idx + 1} Results:")
            print(f"   Average Reward: {success_rate:.3f}")
            print(f"   Success Threshold: {stage['success_threshold']:.3f}")
            print(f"   Status: {'✅ PASSED' if success else '❌ FAILED'}")
            
            # Save evaluation results
            eval_results = {
                'stage': stage_idx + 1,
                'name': stage['name'],
                'success_rate': success_rate,
                'threshold': stage['success_threshold'],
                'success': success,
                'checkpoint_path': checkpoint_path
            }
            
            with open(os.path.join(log_dir, 'evaluation.pkl'), 'wb') as f:
                pickle.dump(eval_results, f)
            
            return success, checkpoint_path, eval_results
        
        except Exception as e:
            print(f"❌ Error in stage {stage_idx + 1}: {e}")
            return False, None, {'error': str(e)}
    
    def evaluate_stage_success(self, runner, stage):
        """Evaluate if stage training was successful"""
        try:
            # Get recent training statistics
            if hasattr(runner, 'log_dict') and runner.log_dict:
                # Try different possible reward keys
                reward_keys = ['Episode/mean_reward', 'Train/mean_reward', 'episode_reward']
                
                for key in reward_keys:
                    if key in runner.log_dict and runner.log_dict[key]:
                        recent_rewards = runner.log_dict[key][-50:]  # Last 50 episodes
                        if recent_rewards:
                            avg_reward = np.mean(recent_rewards)
                            return avg_reward
            
            # Fallback: return a default value
            print("⚠️ Could not determine training performance")
            return 0.0
            
        except Exception as e:
            print(f"⚠️ Error evaluating success: {e}")
            return 0.0
    
    def train_curriculum(self):
        """Train through all curriculum stages"""
        print(f"🎓 Starting curriculum learning: {self.exp_name}")
        print(f"📚 Training through {len(self.stages)} stages")
        
        checkpoint = None
        all_results = []
        
        for i, stage in enumerate(self.stages):
            success, new_checkpoint, results = self.train_stage(i, checkpoint)
            all_results.append(results)
            
            if success:
                print(f"✅ Stage {i+1} ({stage['name']}) completed successfully!")
                checkpoint = new_checkpoint
                self.current_stage = i + 1
            else:
                print(f"❌ Stage {i+1} ({stage['name']}) failed")
                # Continue with current checkpoint
                checkpoint = new_checkpoint if new_checkpoint else checkpoint
        
        # Save final results
        final_results = {
            'experiment_name': self.exp_name,
            'completed_stages': self.current_stage,
            'total_stages': len(self.stages),
            'final_checkpoint': checkpoint,
            'stage_results': all_results,
            'timestamp': datetime.now().isoformat()
        }
        
        results_dir = f"logs/{self.exp_name}_curriculum"
        os.makedirs(results_dir, exist_ok=True)
        results_path = os.path.join(results_dir, 'curriculum_results.pkl')
        
        with open(results_path, 'wb') as f:
            pickle.dump(final_results, f)
        
        print(f"\n🎉 Curriculum training completed!")
        print(f"📊 Completed {self.current_stage}/{len(self.stages)} stages")
        print(f"🏆 Final model: {checkpoint}")
                
        return checkpoint, final_results

# Convenience function for easy import
def run_curriculum_training(exp_name, num_envs=2048, total_iterations=1200):
    """Convenience function to run curriculum training"""
    # Import here to avoid circular imports
    from go2_train_obstacle import get_cfgs, get_train_cfg
    
    print(f"🔧 Setting up curriculum training: {exp_name}")
    
    # Get base configurations
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(exp_name, total_iterations)
    train_cfg['runner']['num_envs'] = num_envs
    
    base_cfg = {
        'env_cfg': env_cfg,
        'obs_cfg': obs_cfg,
        'reward_cfg': reward_cfg,
        'command_cfg': command_cfg,
        'train_cfg': train_cfg
    }
    
    trainer = CurriculumTrainer(base_cfg, exp_name)
    return trainer.train_curriculum()

# Usage
if __name__ == "__main__":
    print("🚀 Running curriculum learning directly...")
    
    gs.init(logging_level="warning")
    
    # Run with test parameters
    final_checkpoint, results = run_curriculum_training("test_curriculum", num_envs=1024)
    
    print(f"\n🏁 Direct execution completed!")
    print(f"Final checkpoint: {final_checkpoint}")