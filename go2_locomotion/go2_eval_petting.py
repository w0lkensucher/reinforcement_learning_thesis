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
from importlib import metadata

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

from go2_env_petting import Go2PettingEnv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2-petting")
    parser.add_argument("--ckpt", type=int, default=100)

    # evaluation petting options
    parser.add_argument("--manual_petting", action='store_true',
                        help="Manually trigger petting every N steps")
    parser.add_argument("--petting_interval", type=int, default=200,
                        help="Steps between manual petting events")
    parser.add_argument("--petting_duration", type=int, default=100,
                        help="Duration of manual petting events in steps")
    parser.add_argument("--petting_frequency", type=float, default= 0.01, # 0.01 = 1% chance per step
                        help="Probability of petting event per step (if not manual)")
    parser.add_argument("--eval_steps", type=int, default=1000,
                        help="Interval (in steps) to print petting statistics")
    
    # Display options
    parser.add_argument("--stats_interval", type=int, default=500,
                        help="Print stats every N steps")
    parser.add_argument("--silent", action='store_true',
                        help="Suppress detailed output")
    
    args = parser.parse_args()

    gs.init()

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(
        open(f"logs/{args.exp_name}/cfgs.pkl", "rb")
    )
    
    # Disable training rewards for pure behavior evaluation
    reward_cfg["reward_scales"] = {}

    # Override petting settings for evaluation
    if args.manual_petting:
        env_cfg["petting_probability"] = 0.0  # Disable automatic petting
        if not args.silent:
            print(f"🖐️ Manual petting mode: every {args.petting_interval} steps for {args.petting_duration} steps")
    else:
        env_cfg["petting_probability"] = args.petting_frequency
        if not args.silent:
            print(f"🤖 Automatic petting mode: {args.petting_frequency*100:.1f}% chance per step")

    env = Go2PettingEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    runner.load(resume_path)

    policy = runner.get_inference_policy(device=gs.device)

    obs, _ = env.reset()
    
    with torch.no_grad():
        while True:
            actions = policy(obs)
            obs, rews, dones, infos = env.step(actions)


if __name__ == "__main__":
    main()