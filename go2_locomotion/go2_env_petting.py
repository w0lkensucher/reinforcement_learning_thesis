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
            self.gesture_duration = 350  # 7 seconds at 50Hz
            self.gesture_cooldown = torch.zeros(self.num_envs, device=self.device, dtype=torch.int)
            self.cooldown_duration = 100  # 2 seconds cooldown
            
            # Track when to resume normal behavior after petting/gesture
            self.resume_normal_behavior = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
            self.falling_edge = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

            # Detection parameters
            self.gentle_speed_threshold = env_cfg.get('gentle_speed_threshold', 0.3)  # Must be moving slowly
            self.gentle_press_range = env_cfg.get('gentle_press_range', [0.02, 0.08])  # 2-8cm push down
            self.gentle_vel_threshold = env_cfg.get('gentle_vel_threshold', 0.1)  # Downward velocity

            # petting parameters
            self.petting_probability = env_cfg.get('petting_probability', 0.005) # Chance to start petting (0.5%)
            self.petting_force_timer = torch.zeros(self.num_envs, device=self.device, dtype=torch.int)
            self.is_being_petted = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
            self.petting_duration = env_cfg.get('petting_duration', 50)  # Duration of petting force application (1s at 50Hz)

            self.manual_petting_active = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
            self.manual_petting_timer = torch.zeros(self.num_envs, device=self.device, dtype=torch.int)
            
            # Debug counter
            self.global_step = 0

            self._setup_scene(show_viewer)
            self._setup_robot()

            if self.petting_probability > 0 or self.env_cfg.get("manual_petting", False):
                self._setup_petting_hands()  # Add hands AFTER robot to avoid DOF index shifting
                
            self._setup_buffers()
            self._build_scene_and_setup()


    def _setup_scene(self, show_viewer=False):
        """Override to add optional camera to the scene"""
        super()._setup_scene(show_viewer)
        # Note: Petting hands are added AFTER robot in __init__ to preserve DOF indices
        
        # Add optional camera BEFORE scene.build()
        if self.env_cfg.get("visualize_camera", False):
            self.cam = self.scene.add_camera(
                # res=(960, 540),
                res=(1920, 1080),
                pos=(4.0, 0.0, 4.0),
                lookat=(0, 0, 1.0),
                fov=30,
                GUI=True
            )


    def _setup_buffers(self):
        super()._setup_buffers()
        
        # Add link contact forces buffer for head pressure detection
        self.link_contact_forces = torch.zeros(
            (self.num_envs, self.robot.n_links, 3), device=self.device, dtype=gs.tc_float
        )
        
        # Find head link index for contact detection
        # Note: With default merge_fixed_links, Head_lower is merged into base
        self.head_link_index = None
        for link in self.robot.links:
            if link.name == "Head_lower":
                self.head_link_index = link.idx - self.robot.link_start
                print(f"✓ Using Head_lower link for head contact detection at index {self.head_link_index}")
                break
            elif link.name == "base":
                self.head_link_index = link.idx - self.robot.link_start
                print(f"✓ Using base link for head contact detection at index {self.head_link_index} (Head_lower merged)")
                break
        
        if self.head_link_index is None:
            print("ERROR: Head_lower link not found! Check URDF and links_to_keep config!")
            self.head_link_index = 0  # Fallback


    def _setup_petting_hands(self):
        """Setup physical hand collision objects for realistic petting contact simulation"""
        # Create ONE hand entity - Genesis automatically creates one instance per environment
        self.petting_hand = self.scene.add_entity(
            morph=gs.morphs.Sphere(
                radius=0.05,  # 5cm radius hand
                pos=(0.0, 0.0, 2.0),  # Start high up to avoid initial collision
                fixed=False,  # Allow it to move
                visualization=True,  # Enable visual rendering
                collision=True,  # Enable collision detection
            ),
            material=gs.materials.Rigid(
                rho=200.0,  # Light hand for gentle contact (~100g total mass, ~1N force)
                friction=0.8,
            ),
        )
        
        # Petting parameters
        self.hand_rest_height = 0.15  # Height above head when not petting (reduced to slow fall)
        self.hand_petting_height = 0.02  # Height above head when petting (slight contact)
        
        # Head position offset from base (Go2-specific: head is forward and slightly up)
        # This is in the robot's local frame and will be rotated with base orientation
        self.head_offset_local = torch.tensor(
            [0.25, 0.0, 0.15],  # [x_forward, y_lateral, z_up] in robot frame
            device=self.device
        )


    def _get_head_world_position(self):
        """Compute head position in world coordinates accounting for robot rotation"""
        # Convert quaternion to rotation matrix for all environments
        # base_quat is [num_envs, 4] in format [w, x, y, z]
        qw, qx, qy, qz = self.base_quat[:, 0], self.base_quat[:, 1], self.base_quat[:, 2], self.base_quat[:, 3]
        
        # Rotation matrix from quaternion (3x3 for each env)
        # Using standard quaternion to rotation matrix conversion
        rot_mat = torch.zeros((self.num_envs, 3, 3), device=self.device)
        
        rot_mat[:, 0, 0] = 1 - 2*(qy*qy + qz*qz)
        rot_mat[:, 0, 1] = 2*(qx*qy - qz*qw)
        rot_mat[:, 0, 2] = 2*(qx*qz + qy*qw)
        
        rot_mat[:, 1, 0] = 2*(qx*qy + qz*qw)
        rot_mat[:, 1, 1] = 1 - 2*(qx*qx + qz*qz)
        rot_mat[:, 1, 2] = 2*(qy*qz - qx*qw)
        
        rot_mat[:, 2, 0] = 2*(qx*qz - qy*qw)
        rot_mat[:, 2, 1] = 2*(qy*qz + qx*qw)
        rot_mat[:, 2, 2] = 1 - 2*(qx*qx + qy*qy)
        
        # Rotate head offset from local frame to world frame: R * offset_local
        head_offset_world = torch.bmm(rot_mat, self.head_offset_local.unsqueeze(0).expand(self.num_envs, -1).unsqueeze(2)).squeeze(2)
        
        # Add to base position to get world position
        head_world_pos = self.base_pos + head_offset_world
        return head_world_pos


    # petting specific methods
    def _detect_head_pressure(self):
        """Detect gentle pressure on the head using contact forces on base link (where Head_lower is merged)"""
        if self.head_link_index is None:
            return torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        
        # Get contact force on the base link (Head_lower merged into base)
        head_contact_force = self.link_contact_forces[:, self.head_link_index, :]  # Shape: [num_envs, 3]
        head_force_z = torch.abs(head_contact_force[:, 2])  # Use absolute value - magnitude matters, not direction
        
        # DEBUG: Print position-based detection info when petting is active
        if self.is_being_petted.any():
            petted_envs = self.is_being_petted.nonzero(as_tuple=False).flatten()
            # Always print Env 0 if it's being petted, otherwise print first petted env
            debug_env = 0 if self.is_being_petted[0] else petted_envs[0].item()
            
            # Get hand position to check if it's actually at the head
            hand_qpos = self.petting_hand.get_qpos()[debug_env]  # [7]: xyz + quat
            hand_pos = hand_qpos[:3]
            head_pos = self._get_head_world_position()[debug_env]
            hand_head_distance = torch.norm(hand_pos - head_pos).item()
            
            # Position-based detection criteria
            in_contact = hand_head_distance < 0.08
            height_ok = self.base_pos[debug_env, 2] > 0.20
            roll = self.base_radians[debug_env, 0].item() * 180 / 3.14159
            pitch = self.base_radians[debug_env, 1].item() * 180 / 3.14159
            roll_thresh = self.env_cfg.get("termination_if_roll_greater_than", 45)
            pitch_thresh = self.env_cfg.get("termination_if_pitch_greater_than", 45)
            upright = (abs(roll) < roll_thresh) and (abs(pitch) < pitch_thresh)
            
            # print(f"[DEBUG] Env {debug_env}: "
            #       f"Dist={hand_head_distance:.3f}m (contact={in_contact}, thresh=0.08m), "
            #       f"Hand[{hand_pos[0].item():.2f}, {hand_pos[1].item():.2f}, {hand_pos[2].item():.2f}] "
            #       f"Head[{head_pos[0].item():.2f}, {head_pos[1].item():.2f}, {head_pos[2].item():.2f}], "
            #       f"Height={self.base_pos[debug_env, 2].item():.2f}m (ok={height_ok}), "
            #       f"Roll={roll:.1f}° Pitch={pitch:.1f}° (upright={upright}), "
            #       f"Timer={self.petting_force_timer[debug_env].item()}, Cooldown={self.gesture_cooldown[debug_env].item()}, "
            #       f"DETECTED={in_contact & height_ok & upright & self.is_being_petted[debug_env]}")
        
        # Since collision forces aren't working, use POSITION-BASED detection
        # Get hand positions for all environments
        hand_qpos_all = self.petting_hand.get_qpos()  # [num_envs, 7]
        hand_positions = hand_qpos_all[:, :3]  # [num_envs, 3] - xyz positions
        head_positions = self._get_head_world_position()  # [num_envs, 3]
        
        # Distance between hand and head
        hand_head_distance = torch.norm(hand_positions - head_positions, dim=1)  # [num_envs]
        
        # Conditions for petting detection:
        # 1. Hand within contact range (sphere radius + small margin)
        in_contact = hand_head_distance < 0.08  # 5cm radius + 3cm margin
        
        # 2. Not in cooldown
        not_in_cooldown = self.gesture_cooldown == 0
        
        # 3. Robot at reasonable height (not on ground or fallen)
        at_standing_height = self.base_pos[:, 2] > 0.20
        
        # 4. Robot upright (not fallen over)
        roll_thresh = self.env_cfg.get("termination_if_roll_greater_than", 45) * 3.14159 / 180
        pitch_thresh = self.env_cfg.get("termination_if_pitch_greater_than", 45) * 3.14159 / 180
        robot_upright = (torch.abs(self.base_radians[:, 0]) < roll_thresh) & \
                        (torch.abs(self.base_radians[:, 1]) < pitch_thresh)
        
        # 5. Petting is actually active (hand should be lowered)
        petting_active = self.is_being_petted
        
        pressure_touch = (in_contact & 
                        not_in_cooldown & 
                        at_standing_height &
                        robot_upright &
                        petting_active)
        
        return pressure_touch
    

    def _check_head_petting(self):
        """Checking for head petting gestures."""
        # Skip petting detection if no hands exist (petting_probability=0)
        if not hasattr(self, 'petting_hand'):
            return
        
        # Track falling edge for resuming normal behavior
        if not hasattr(self, '_prev_head_touched'):
            self._prev_head_touched = torch.zeros_like(self.head_touched)
            
        self.head_touched = self._detect_head_pressure()

        self.falling_edge = (~self.head_touched) & self._prev_head_touched

        # Only trigger gesture on rising edge (when petting starts this step)
        touched_envs = self.head_touched
        # Rising edge: was not touched last step, now touched
        if not hasattr(self, '_prev_head_touched'):
            self._prev_head_touched = torch.zeros_like(touched_envs)
        rising_edge = touched_envs & (~self._prev_head_touched)
        self.gesture_timer = torch.where(rising_edge,
                                         torch.full_like(self.gesture_timer, self.gesture_duration),
                                         self.gesture_timer)
        self.gesture_cooldown = torch.where(rising_edge,
                                            torch.full_like(self.gesture_cooldown, self.cooldown_duration),
                                            self.gesture_cooldown)
        self._prev_head_touched = touched_envs.clone()
        

    # Test limits of movement
    def _get_petting_gesture_actions(self):
        """Generate friendly petting response gesture - SMALL ADDITIVE adjustments to learned actions"""
        # Start with learned actions (will be blended in step() function)
        gesture_adjustments = torch.zeros_like(self.actions)
        
        # Create gentle wave pattern
        wave_phase = (self.gesture_timer.float() / 20.0) % (2 * torch.pi)
        wave_amplitude = 0.3 
        
        # Very subtle "happy" movement - just small additive wiggles
        # These are ADDED to learned actions, not replacements
        gesture_adjustments[:, 3] = wave_amplitude * 0.3 * torch.sin(wave_phase)       # FL_hip
        gesture_adjustments[:, 4] = wave_amplitude * 0.2 * torch.sin(wave_phase)       # FL_thigh
        gesture_adjustments[:, 5] = wave_amplitude * 0.2 * torch.cos(wave_phase)       # FL_calf
        
        gesture_adjustments[:, 0] = wave_amplitude * 0.3 * torch.sin(wave_phase + 1.57)   # FR_hip
        gesture_adjustments[:, 1] = wave_amplitude * 0.2 * torch.sin(wave_phase + 1.57)   # FR_thigh
        gesture_adjustments[:, 2] = wave_amplitude * 0.2 * torch.cos(wave_phase + 1.57)   # FR_calf
        
        # Very subtle "tail wagging" with rear hips only
        tail_wag = wave_amplitude * 0.3 * torch.sin(wave_phase * 2)
        gesture_adjustments[:, 6] = tail_wag     # RL_hip
        gesture_adjustments[:, 9] = -tail_wag    # RR_hip (opposite)
        # Rear legs thigh/calf: no adjustment, let learned actions handle stability

        # After gesture ends, bias actions toward default joint angles
        if hasattr(self, "env_cfg") and "default_joint_angles" in self.env_cfg and "joint_names" in self.env_cfg:
            # Build tensor of default joint angles in correct order
            joint_names = self.env_cfg["joint_names"]
            default_angles = self.env_cfg["default_joint_angles"]
            default_angles_tensor = torch.tensor(
                [default_angles[name] for name in joint_names],
                device=self.device
            ).unsqueeze(0).expand(self.num_envs, -1)
            # If gesture_timer is zero, encourage return to default
            post_gesture = (self.gesture_timer == 0).unsqueeze(1)
            # Blend: if gesture_timer==0, set adjustment to move toward default
            # (Assume self.dof_pos is [num_envs, num_joints])
            if hasattr(self, "dof_pos"):
                to_default = default_angles_tensor - self.dof_pos
                # Use a moderate gain to avoid abrupt jumps
                gain = 0.5
                gesture_adjustments = torch.where(
                    post_gesture,
                    gain * to_default,
                    gesture_adjustments
                )
        return gesture_adjustments


    def _move_hand_to_position(self, env_id, target_pos, zero_velocity=True):
        """Helper function to move hand sphere to target position using qpos
        
        Args:
            env_id: Environment index
            target_pos: 3D position tensor [x, y, z]
            zero_velocity: If True, lock hand in place; if False, allow gravity/physics
        """
        # Set hand position using qpos (7-vector: xyz + quaternion wxyz)
        # For FREE joint: [x, y, z, quat_w, quat_x, quat_y, quat_z]
        identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        qpos = torch.cat([target_pos, identity_quat])  # Shape: [7]
        
        # Specify which environment to update (single entity, multiple instances)
        self.petting_hand.set_qpos(
            qpos, 
            envs_idx=[env_id], 
            zero_velocity=zero_velocity
        )


    def _apply_random_petting_forces(self):
        """Apply random petting by moving hand colliders down to touch the head"""
        
        if not hasattr(self, 'petting_hand'):
            return
        
        # Random chance of starting new petting interaction
        # Note: Don't check gesture_cooldown here - petting should be able to start anytime
        # The cooldown only prevents triggering NEW gestures, not petting itself
        start_petting = (torch.rand(self.num_envs, device=self.device) < self.petting_probability) & \
                       (self.petting_force_timer == 0)
        
        # Update petting timers
        self.petting_force_timer = torch.where(
            start_petting,
            torch.full_like(self.petting_force_timer, self.petting_duration),
            torch.clamp(self.petting_force_timer - 1, 0, self.petting_duration)
        )

        self.is_being_petted = self.petting_force_timer > 0
        
        # Get dynamically computed head positions (tracks rotation)
        head_world_positions = self._get_head_world_position()
        
        # Move hands down for petting, up when not petting
        for env_id in range(self.num_envs):
            current_head_pos = head_world_positions[env_id]
            
            if self.is_being_petted[env_id]:
                # During petting: Position hand near head for contact detection
                # Use 5cm offset (within detection threshold of 8cm) and lock in place
                target_pos = current_head_pos + torch.tensor([0.0, 0.0, 0.05], device=self.device)
                self._move_hand_to_position(env_id, target_pos, zero_velocity=True)
            else:
                # Keep hand well above head when not petting
                target_pos = current_head_pos + torch.tensor([0.0, 0.0, self.hand_rest_height], device=self.device)
                self._move_hand_to_position(env_id, target_pos, zero_velocity=True)


    def _apply_manual_petting(self):
        """Apply manual petting (for testing/debugging)"""
        
        if not hasattr(self, 'petting_hand'):
            return
        
        # Decrease manual petting timers
        self.manual_petting_timer = torch.clamp(self.manual_petting_timer - 1, 0, None)
        active_envs = self.manual_petting_active & (self.manual_petting_timer > 0)
        self.is_being_petted = active_envs
        self.manual_petting_active = self.manual_petting_active & (self.manual_petting_timer > 0)

        # Get dynamically computed head positions (tracks rotation)
        head_world_positions = self._get_head_world_position()

        # Move hands for manual petting
        for env_id in range(self.num_envs):
            current_head_pos = head_world_positions[env_id]
            
            if self.is_being_petted[env_id]:
                # Position hand close to head, allow physics for realistic contact
                target_pos = current_head_pos + torch.tensor([0.0, 0.0, 0.08], device=self.device)
                self._move_hand_to_position(env_id, target_pos, zero_velocity=False)
            else:
                # Keep hand well above when not petting
                target_pos = current_head_pos + torch.tensor([0.0, 0.0, self.hand_rest_height], device=self.device)
                self._move_hand_to_position(env_id, target_pos, zero_velocity=True)




    def trigger_manual_petting(self, env_id=0, duration_steps=50):
        """Externally trigger petting for specific environments (for testing)"""
        if env_id < self.num_envs:
            self.manual_petting_timer[env_id] = duration_steps
            self.manual_petting_active[env_id] = True

    def _update_robot_state(self):
        """Override to also update contact forces for head pressure detection"""
        super()._update_robot_state()
        # Update link contact forces
        self.link_contact_forces[:] =  self.robot.get_links_net_contact_force().detach().clone().to(
            device=self.device,
            dtype=gs.tc_float
            )

    # Reward functions for petting
    def _reward_petting_response(self):
        """Reward for appropriate gesture response to petting"""
        gesture_active = self.gesture_timer > 0
        # Reward proportional to gesture progress
        progress = (self.gesture_duration - self.gesture_timer) / self.gesture_duration
        return torch.where(gesture_active, progress, torch.zeros_like(progress))


    def _reward_petting_stability(self):
        """Reward for maintaining stability during petting gestures"""
        # Reward stability during gesture
        ang_vel_magnitude = torch.norm(self.base_ang_vel, dim=1)
        height_deviation = torch.abs(self.base_pos[:, 2] - self.reward_cfg.get('base_height_target', 0.35))
        
        stability = torch.exp(-(ang_vel_magnitude + height_deviation * 5))

        gesture_active = (self.gesture_timer.float() / self.gesture_duration).clamp(0, 1)
        stability_reward = stability * 0.1 * gesture_active
        
        return stability_reward


    def _reward_flexible_height(self):
        """Petting-specific height control - allow sitting, lying"""
        base_height = self.base_pos[:, 2]
        
        # Different target heights for different behaviors
        if (self.gesture_timer > 0).any() or self.head_touched.any():
            # During gestures, allow more height variation
            tolerance = 0.4
            target_height = self.reward_cfg.get("base_height_target", 0.35)  # Slightly lower for better petting access
        else:
            # Normal standing posture
            tolerance = 0.1
            target_height = 0.42
        
        height_error = torch.abs(base_height - target_height)
        return torch.exp(-10 * torch.clamp(height_error - tolerance, min=0.0))


    def step(self, actions):

        """Petting-specific step logic"""
        self.global_step += 1
        
        # By default, allow normal behavior
        self.resume_normal_behavior[:] = False
        
        if self.manual_petting_active.any():
            # Apply manual petting (for testing/debugging)
            self._apply_manual_petting()
        else:
            # Apply random petting
            self._apply_random_petting_forces()

            # Check for head petting
            self._check_head_petting()
        
        # Add gesture adjustments to learned actions (not replace!)
        active_gesture = self.gesture_timer > 0
        # Resume normal behavior if hand was removed (falling edge) and gesture finished
        self.resume_normal_behavior = self.falling_edge & (self.gesture_timer == 0)
        if active_gesture.any():
            gesture_adjustments = self._get_petting_gesture_actions()
            # ADD small adjustments to learned actions for gesture environments
            actions = actions + torch.where(active_gesture.unsqueeze(1), gesture_adjustments, torch.zeros_like(gesture_adjustments))
        
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