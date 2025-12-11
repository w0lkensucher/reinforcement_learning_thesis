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
            self.gentle_speed_threshold = env_cfg.get('gentle_speed_threshold', 0.3)  # Must be moving slowly
            self.gentle_press_range = env_cfg.get('gentle_press_range', [0.02, 0.08])  # 2-8cm push down
            self.gentle_vel_threshold = env_cfg.get('gentle_vel_threshold', 0.1)  # Downward velocity

            # petting parameters
            self.petting_probability = env_cfg.get('petting_probability', 0.005) # Chance to start petting (0.5%)
            self.petting_force_timer = torch.zeros(self.num_envs, device=self.device, dtype=torch.int)
            self.is_being_petted = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
            self.petting_duration = env_cfg.get('petting_duration', 50)  # Duration of petting force application (1s at 50Hz)

            self._setup_scene(show_viewer)
            self._setup_robot()
            self._setup_buffers()
            self._build_scene_and_setup()


    # petting specific methods
    def _detect_head_pressure(self):
            # Normal target height
            target_height = self.reward_cfg.get("base_height_target", 0.35)
            current_height = self.base_pos[:, 2]
            
            # Calculate height reduction from normal
            height_reduction = target_height - current_height
            
            # Get vertical velocity (negative = being pushed down)
            vertical_velocity = self.base_lin_vel[:, 2]
            gentle_downward_vel = -vertical_velocity  # Positive = being pushed down
            
            # Conditions for gentle head pressure:
            # 1. Height slightly reduced (2-8cm below normal)
            gentle_height_reduction = (height_reduction >= self.gentle_press_range[0]) & (height_reduction <= self.gentle_press_range[1])
            
            # 2. Gentle downward velocity (being pressed down)
            gentle_downward_pressure = gentle_downward_vel > self.gentle_vel_threshold
            
            # 3. Robot relatively stationary (not jumping/falling)
            robot_speed = torch.norm(self.base_lin_vel[:, :2], dim=1)
            is_stationary = robot_speed < self.gentle_speed_threshold
            
            # 4. Not in cooldown
            not_in_cooldown = self.gesture_cooldown == 0
            
            # 5. Robot relatively stable (not tilted)
            roll_pitch_magnitude = torch.norm(self.base_radians[:, :2], dim=1)
            is_stable = roll_pitch_magnitude < 10.0  # Less than 10 degrees tilt
            
            pressure_touch = (gentle_height_reduction & 
                            gentle_downward_pressure & 
                            is_stationary & 
                            not_in_cooldown & 
                            is_stable)
            
            return pressure_touch
    

    def _check_head_petting(self):
        """Checking for head petting gestures."""
        pressure_touches = self._detect_head_pressure()
        
        self.head_touched = pressure_touches

        # Start gesture timer and cooldown for touched environments
        touched_envs = self.head_touched
        self.gesture_timer = torch.where(touched_envs, 
                                    torch.full_like(self.gesture_timer, self.gesture_duration),
                                    self.gesture_timer)
        self.gesture_cooldown = torch.where(touched_envs,
                                        torch.full_like(self.gesture_cooldown, self.cooldown_duration),
                                        self.gesture_cooldown)
        

    def _get_petting_gesture_actions(self):
        """Generate friendly petting response gesture (tail wag + head movement)"""
        gesture_actions = torch.zeros_like(self.actions)
        
        # Create gentle wave pattern
        wave_phase = (self.gesture_timer.float() / 20.0) % (2 * 3.14159)
        wave_amplitude = 0.3
        
        # Gentle "happy" front leg movement (like excited stepping)
        gesture_actions[:, 3] = wave_amplitude * 0.5 * torch.sin(wave_phase)           # FL_hip
        gesture_actions[:, 4] = 0.8 + wave_amplitude * 0.3 * torch.sin(wave_phase)    # FL_thigh
        gesture_actions[:, 5] = -1.5 + wave_amplitude * 0.3 * torch.cos(wave_phase)   # FL_calf
        
        gesture_actions[:, 0] = wave_amplitude * 0.5 * torch.sin(wave_phase + 1.57)   # FR_hip
        gesture_actions[:, 1] = 0.8 + wave_amplitude * 0.3 * torch.sin(wave_phase + 1.57)  # FR_thigh
        gesture_actions[:, 2] = -1.5 + wave_amplitude * 0.3 * torch.cos(wave_phase + 1.57) # FR_calf
        
        # "Tail wagging" with rear legs (gentle side-to-side)
        tail_wag = wave_amplitude * 0.4 * torch.sin(wave_phase * 2)  # Faster tail movement
        gesture_actions[:, 6] = tail_wag    # RL_hip
        gesture_actions[:, 7] = 1.0         # RL_thigh (stable)
        gesture_actions[:, 8] = -1.5        # RL_calf (stable)
        
        gesture_actions[:, 9] = -tail_wag   # RR_hip (opposite direction)
        gesture_actions[:, 10] = 1.0        # RR_thigh (stable)
        gesture_actions[:, 11] = -1.5       # RR_calf (stable)
        
        return gesture_actions


    def _apply_random_petting_forces(self):
        """Apply random petting forces using the petting objects"""
        # Random chance of starting new petting interaction
        start_petting = (torch.rand(self.num_envs, device=self.device) < self.petting_probability) & \
                       (self.petting_force_timer == 0) & \
                       (self.gesture_cooldown == 0)
        
        # Update petting timers
        self.petting_force_timer = torch.where(
            start_petting,
            torch.full_like(self.petting_force_timer, self.petting_duration),
            torch.clamp(self.petting_force_timer - 1, 0, self.petting_duration)
        )
        
        self.is_being_petted = self.petting_force_timer > 0
        
        if self.is_being_petted.any():
            # Method 1: Modify base velocity directly (simulates gentle pressure)
            downward_velocity = torch.zeros_like(self.base_lin_vel)
            downward_velocity[self.is_being_petted, 2] = -0.12  # 0.12 m/s downward
            
            # Apply the velocity change
            self.base_lin_vel += downward_velocity * 0.1  # Gradual application
            
            # Method 2: Slightly lower the robot height (simulates compression)
            height_reduction = torch.zeros_like(self.base_pos[:, 2])
            height_reduction[self.is_being_petted] = -0.03  # 3cm lower
            self.base_pos[self.is_being_petted, 2] += height_reduction[self.is_being_petted] * 0.1


    def trigger_manual_petting(self, env_id=0, duration_steps=50):
        """Externally trigger petting for specific environments (for testing)"""
        if env_id < self.num_envs:
            self.petting_force_timer[env_id] = duration_steps
            self.is_being_petted[env_id] = True

    # Reward functions for petting
    def _reward_petting_response(self):
        """Reward for appropriate gesture response to petting"""
        gesture_reward = torch.where(
            self.gesture_timer > 0,
            torch.full_like(self.head_touched, 0.2, dtype=torch.float),
            torch.zeros_like(self.head_touched, dtype=torch.float)
        )
        return gesture_reward


    def _reward_petting_stability(self):
        """Reward for maintaining stability during petting gestures"""
        # Reward stability during gesture
        ang_vel_magnitude = torch.norm(self.base_ang_vel, dim=1)
        height_deviation = torch.abs(self.base_pos[:, 2] - self.reward_cfg.get('base_height_target', 0.35))
        
        stability = torch.exp(-(ang_vel_magnitude + height_deviation * 5))
        
        stability_reward = torch.where(
            self.gesture_timer > 0,
            stability * 0.1,
            torch.zeros_like(stability)
        )
        
        return stability_reward
    

    def _reward_calm_behavior(self):
        """Reward for calm, gentle behavior when not being petted"""
        not_in_gesture = self.gesture_timer == 0
        
        # Reward low movement when not in gesture
        movement_penalty = torch.norm(self.base_lin_vel, dim=1)
        angular_penalty = torch.norm(self.base_ang_vel, dim=1)
        
        calmness = torch.exp(-(movement_penalty + angular_penalty))
        
        return torch.where(not_in_gesture, calmness * 0.05, torch.zeros_like(calmness))


    def _reward_flexible_height(self):
        """Petting-specific height control - allow sitting, lying"""
        base_height = self.base_pos[:, 2]
        
        # Different target heights for different behaviors
        if (self.gesture_timer > 0).any():
            # During gestures, allow more height variation
            tolerance = 0.4
            target_height = 0.35  # Slightly lower for better petting access
        elif self.head_touched.any():
            # When being petted, allow sitting/lying
            tolerance = 0.6
            target_height = 0.25  # Allow lower postures
        else:
            # Normal standing posture
            tolerance = 0.1
            target_height = 0.42
        
        height_error = torch.abs(base_height - target_height)
        return -torch.where(height_error < tolerance, 
                            height_error * 0.2,  # Very gentle penalty
                            height_error * 1.0)  # Moderate penalty
    
    
    def _reward_no_fall(self):
        # Negative reward if roll or pitch exceeds threshold
        roll = torch.abs(self.base_radians[:, 0])
        pitch = torch.abs(self.base_radians[:, 1])
        roll_thresh = self.env_cfg.get("termination_if_roll_greater_than", 45)
        pitch_thresh = self.env_cfg.get("termination_if_pitch_greater_than", 45)
        fallen = (roll > roll_thresh) | (pitch > pitch_thresh)
        return torch.where(fallen, torch.full_like(roll, -2.0), torch.zeros_like(roll))

    def step(self, actions):
        """Petting-specific step logic"""
        # Apply random petting forces
        self._apply_random_petting_forces()

        # Check for head petting
        self._check_head_petting()
        
        # Override actions if gesture is active
        active_gesture = self.gesture_timer > 0
        if active_gesture.any():
            gesture_actions = self._get_petting_gesture_actions()
            # Apply gesture only to environments where timer is active
            actions = torch.where(active_gesture.unsqueeze(1), gesture_actions, actions)
        
        # Update timers
        self.gesture_timer = torch.clamp(self.gesture_timer - 1, 0, self.gesture_duration)
        self.gesture_cooldown = torch.clamp(self.gesture_cooldown - 1, 0, self.cooldown_duration)
        
        # Execute actions using shared method
        self._execute_actions(actions)
        
        # Update robot state using shared method
        self._update_robot_state()
        
        # Resample commands (petting uses slower commands)
        envs_idx = (
            (self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0)
            .nonzero(as_tuple=False)
            .reshape((-1,))
        )
        self._resample_commands(envs_idx)
        
        # Check termination using shared method
        self._check_termination()
        
        # Reset environments
        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).reshape((-1,)))
        
        # Compute petting-specific rewards
        self._compute_rewards()
        
        # Compute petting-specific observations
        self._compute_observations()
        
        # Update for next step (shared)
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        
        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras


    def _compute_observations(self):
        """Compute observations including petting-specific data"""
        gesture_active = (self.gesture_timer > 0).float()
        touch_detected = self.head_touched.float()
        
        self.obs_buf = torch.cat([
            self.base_ang_vel * self.obs_scales["ang_vel"],  # 3
            self.projected_gravity,  # 3
            self.commands * self.commands_scale,  # 3
            (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 12
            self.dof_vel * self.obs_scales["dof_vel"],  # 12
            self.actions,  # 12
            gesture_active.unsqueeze(1),  # 1: gesture active signal
            touch_detected.unsqueeze(1),  # 1: touch detected signal
        ], axis=-1)


    def get_observations(self):
        """Get navigation observations"""
        self.extras["observations"]["critic"] = self.obs_buf
        return self.obs_buf, self.extras


    def _compute_rewards(self):
        """Petting-specific reward computation"""
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        self.episode_sums['reward'] += self.rew_buf

    def _resample_commands(self, envs_idx):
        """Petting-specific command resampling (slower, gentler movements)"""
        # Use gentler command ranges for petting
        petting_ranges = {
            "lin_vel_x_range": [-0.5, 0.5],  # Slower movement
            "lin_vel_y_range": [-0.3, 0.3],  # Less lateral movement  
            "ang_vel_range": [-0.5, 0.5]     # Slower turning
        }
        
        self.commands[envs_idx, 0] = torch.rand(len(envs_idx), device=self.device) * \
                                    (petting_ranges["lin_vel_x_range"][1] - petting_ranges["lin_vel_x_range"][0]) + \
                                    petting_ranges["lin_vel_x_range"][0]
        self.commands[envs_idx, 1] = torch.rand(len(envs_idx), device=self.device) * \
                                    (petting_ranges["lin_vel_y_range"][1] - petting_ranges["lin_vel_y_range"][0]) + \
                                    petting_ranges["lin_vel_y_range"][0]
        self.commands[envs_idx, 2] = torch.rand(len(envs_idx), device=self.device) * \
                                    (petting_ranges["ang_vel_range"][1] - petting_ranges["ang_vel_range"][0]) + \
                                    petting_ranges["ang_vel_range"][0]