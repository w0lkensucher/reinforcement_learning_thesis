import os
import sys
if sys.version_info < (3, 12):
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

from tqdm import tqdm
from  contextlib import redirect_stdout, redirect_stderr
from io import StringIO
import torch
import numpy as np
from datetime import datetime

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
    parser.add_argument("--petting_interval", type=int, default=500,
                        help="Steps between manual petting events")
    parser.add_argument("--petting_duration", type=int, default=50,
                        help="Duration of manual petting events in steps")
    parser.add_argument("--petting_frequency", type=float, default= 0.01, # 0.01 = 1% chance per step
                        help="Probability of petting event per step (if not manual)")
    parser.add_argument("--eval_steps", type=int, default=1000,
                        help="Interval (in steps) to print petting statistics")
    parser.add_argument("--record", action='store_true', default=False,
                        help="Record evaluation video")
    
    # Display options
    parser.add_argument("--stats_interval", type=int, default=500,
                        help="Print stats every N steps")
    parser.add_argument("--silent", action='store_true',
                        help="Suppress detailed output")
    
    args = parser.parse_args()

    gs.init(logging_level="warning")

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(
        open(f"logs/{args.exp_name}/cfgs.pkl", "rb")
    )
    
    # Disable training rewards for pure behavior evaluation
    reward_cfg["reward_scales"] = {}
    env_cfg["episode_length_s"] = 60.0

    max_sim_step = int(env_cfg["episode_length_s"] * 50)

    # Override petting settings for evaluation
    if args.manual_petting:
        env_cfg["petting_probability"] = 0.0  # Disable automatic petting
        env_cfg["manual_petting"] = True
        if not args.silent:
            print(f"🖐️ Manual petting mode: every {args.petting_interval} steps for {args.petting_duration} steps")
    else:
        env_cfg["petting_probability"] = args.petting_frequency
        if not args.silent:
            print(f"🤖 Automatic petting mode: {args.petting_frequency*100:.1f}% chance per step")

    if args.silent:
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            env = Go2PettingEnv(
                num_envs=1,
                env_cfg=env_cfg,
                obs_cfg=obs_cfg,
                reward_cfg=reward_cfg,
                command_cfg=command_cfg,
                show_viewer=True,
            )
    else:
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
        if args.record:
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                env.cam.start_recording()

            print("🎥 Recording video...")
        
        with tqdm(total=max_sim_step, 
                desc="🐕 Evaluating Petting", 
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
                file=sys.stdout,
                dynamic_ncols=True) as pbar:
            
            petting_events = 0
            manual_petting_timer = 0

            for step in range(max_sim_step):
                if args.manual_petting:
                    if step % args.petting_interval == 0:
                        manual_petting_timer = args.petting_duration
                        petting_events += 1
                        env.trigger_manual_petting(env_id=0, duration_steps=manual_petting_timer)
                        if not args.silent:
                            print(f"🖐️ Manual petting started at step {step} for {args.petting_duration} steps")
                    
                    if manual_petting_timer > 0:
                        manual_petting_timer -= 1
                    
                    if manual_petting_timer == 0 and step % args.petting_interval == args.petting_duration:
                        if not args.silent:
                            print(f"🖐️ Manual petting ended at step {step}")

                actions = policy(obs)
                obs, rews, dones, infos = env.step(actions)
                env.cam.render()
                
                reset_idx = dones.nonzero(as_tuple=False).squeeze(-1)
                if len(reset_idx) > 0:
                    env.reset_idx(reset_idx)

                pbar.update(1)
                        
        if args.record:
            print("🎥 Stopping recording...")
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                env.cam.stop_recording(save_to_filename=f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.exp_name}.mp4",fps=30)
                    
    print(f"✅ Evaluation complete! Total petting events: {petting_events}")

if __name__ == "__main__":
    main()