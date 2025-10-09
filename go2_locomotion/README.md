# Go2 Locomotion Environment

This directory contains a reinforcement learning environment for training the Go2 quadruped robot to navigate with obstacle avoidance using Genesis simulation.

## Setup Instructions

### 1. Install Dependencies

First, make sure you have Python 3.8+ and CUDA installed (for GPU acceleration).

```bash
# Install basic requirements
pip install -r requirements_go2.txt

# Install Genesis (follow official installation guide)
pip install genesis-world

# IMPORTANT: Make sure you have the correct RSL-RL version
pip uninstall rsl_rl  # Remove old version if exists
pip install rsl-rl-lib==2.2.4
```

### 2. Genesis Assets

The environment requires Genesis URDF assets. Make sure Genesis is properly installed and the following assets are available:
- `urdf/plane/plane.urdf` (ground plane)
- `urdf/go2/urdf/go2.urdf` (Go2 robot model)

These should be automatically available after Genesis installation.

### 3. Running the Training

Navigate to the `go2_locomotion` directory and run:

```bash
cd go2_locomotion

# Basic training (no obstacles, good for testing setup)
python train_go2_obstacles.py -e go2-test --num_envs 64 --max_iterations 10

# Full training with obstacles
python train_go2_obstacles.py -e go2-obstacles --num_envs 2048 --max_iterations 500

# Training with fewer environments (for limited GPU memory)
python train_go2_obstacles.py -e go2-small --num_envs 512 --max_iterations 200
```

### 4. Configuration

The environment can be configured by modifying the `get_cfgs()` function in `train_go2_obstacles.py`:

- **Environment settings**: Episode length, action scaling, termination conditions
- **Obstacle settings**: Density, types, sizes, spacing
- **Reward settings**: Reward weights for different behaviors
- **Command settings**: Velocity ranges for locomotion commands

### 5. Key Features

- **Obstacle Avoidance**: Configurable random obstacles in the environment
- **Locomotion Control**: 12-DOF quadruped locomotion with PD control
- **Reward Shaping**: Multiple reward components for stable locomotion
- **Multi-Environment**: Parallel simulation for efficient training

### 6. Troubleshooting

**Import Errors**: Make sure Genesis and rsl-rl-lib are properly installed
**GPU Issues**: Check CUDA compatibility with PyTorch version
**URDF Errors**: Verify Genesis assets are properly installed

### 7. File Structure

```
go2_locomotion/
├── __init__.py                 # Package initialization
├── go2_env.py                 # Main environment class
├── train_go2_obstacles.py     # Training script
└── logs/                      # Training logs and models (created during training)
    └── [experiment_name]/
        ├── cfgs.pkl          # Saved configurations
        └── model_*.pt        # Trained model checkpoints
```

## Environment Details

- **Observation Space**: 45-dimensional (angular velocity, gravity, commands, joint positions/velocities, actions)
- **Action Space**: 12-dimensional joint position targets
- **Simulation**: 50Hz control frequency, Genesis physics engine
- **Learning**: PPO algorithm with RSL-RL framework