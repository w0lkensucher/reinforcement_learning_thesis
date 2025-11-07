import torch
import genesis as gs
import numpy as np
from go2_env_base import Go2BaseEnv

class Go2PettingEnv(Go2BaseEnv):
   def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False):
        # Initialize base environment FIRST
        super().__init__(num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg)

        self.head_touched = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.gesture_timer = torch.zeros(self.num_envs, device=self.device, dtype=torch.int)
        self.gesture_duration = 100  # 2 seconds at 50Hz
        self.gesture_cooldown = torch.zeros(self.num_envs, device=self.device, dtype=torch.int)
        self.cooldown_duration = 250  # 5 seconds cooldown

        # Detection parameters
        self.head_touch_radius = env_cfg.get('head_touch_radius', 0.2)  # 20cm detection range
        self.gentle_speed_threshold = env_cfg.get('gentle_speed_threshold', 0.3)  # Must be moving slowly
        self.head_height_offset = env_cfg.get('head_height_offset', 0.25)  # Head 25cm above base

        # Height-based detection parameters
        self.use_height_detection = env_cfg.get('use_height_detection', True)
        self.gentle_press_range = env_cfg.get('gentle_press_range', [0.02, 0.08])  # 2-8cm push down
        self.gentle_vel_threshold = env_cfg.get('gentle_vel_threshold', 0.1)  # Downward velocity

        # Add gesture observations to obs buffer
        self.num_obs += 2  # gesture_active and touch_detected
    
        self._setup_scene(show_viewer)
        self._setup_robot()
        self._setup_buffers()
        self._build_scene_and_setup()