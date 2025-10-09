# Reinforcement Learning Thesis

This repository contains various reinforcement learning implementations and experiments for thesis research.

## Contents

### Classic RL Algorithms
- `gridworld.py` - Gridworld environment implementation
- `gridworld_policy_eval.py` - Policy evaluation in gridworld
- `gridworld_policy_iter_action_val.py` - Policy iteration with action values
- `car_rental.py` - Car rental problem environment
- `car_rental_policy_iter.py` - Policy iteration for car rental
- `multi_armed_bandits.py` - Multi-armed bandit implementations

### Go2 Locomotion Environment
- `go2_locomotion/` - Complete Go2 quadruped robot locomotion environment
  - Genesis-based physics simulation
  - Obstacle avoidance capabilities
  - PPO training with RSL-RL
  - See `go2_locomotion/README.md` for detailed setup instructions

## Setup

### For Classic RL Algorithms
```bash
pip install numpy matplotlib
```

### For Go2 Locomotion
```bash
pip install -r requirements_go2.txt
pip install genesis-world
pip install rsl-rl-lib==2.2.4
```

See `go2_locomotion/README.md` for complete setup instructions.

## Usage

### Classic RL
Run individual Python files directly:
```bash
python gridworld_policy_eval.py
python car_rental_policy_iter.py
```

### Go2 Locomotion Training
```bash
cd go2_locomotion
python train_go2_obstacles.py -e my-experiment --num_envs 512 --max_iterations 200
```

## Repository Structure

```
reinforcement_learning_thesis/
├── README.md                           # This file
├── requirements_go2.txt                # Dependencies for Go2 environment
├── gridworld.py                        # Gridworld environment
├── gridworld_policy_eval.py           # Policy evaluation
├── gridworld_policy_iter_action_val.py # Policy iteration
├── car_rental.py                       # Car rental environment
├── car_rental_policy_iter.py          # Car rental policy iteration
├── multi_armed_bandits.py             # Multi-armed bandits
└── go2_locomotion/                     # Go2 locomotion environment
    ├── README.md                       # Detailed setup guide
    ├── __init__.py                     # Package initialization
    ├── go2_env.py                     # Main environment class
    ├── train_go2_obstacles.py         # Training script
    └── logs/                          # Training results (created during training)
```

## Research Focus

This repository explores various aspects of reinforcement learning:

1. **Foundational Algorithms**: Implementation of classic RL algorithms on standard problems
2. **Modern Deep RL**: Advanced locomotion control using modern deep RL techniques
3. **Sim-to-Real**: Genesis simulation environment designed for real robot deployment
4. **Obstacle Navigation**: Complex navigation scenarios with dynamic obstacle avoidance

## Requirements

- Python 3.8+
- NumPy, Matplotlib (for classic RL)
- PyTorch (for Go2 environment)
- Genesis simulation framework (for Go2 environment)
- CUDA-capable GPU (recommended for Go2 training)