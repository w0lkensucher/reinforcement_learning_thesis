"""Regenerate all 'across runs' figures in thesis/figures/ from 5 seeds (1, 8, 42, 24, 77).

Per curriculum/task stage it merges the (resumption) TensorBoard event files of each seed
into one continuous curve and plots all five seeds on a single axis, overwriting the
existing PNGs in place.

Handling rules per stage:
  - walk  : keep steps >= 50 (stage proper); merge multi-file resumes keep-first per step
  - avoid : keep only event files whose max step > 1049 (skip the shared walk base),
            and keep steps >= 1049 (resume point); merge keep-first
  - jump  : same as avoid (steps >= 1049)
  - stand (nav + petting), gestures: all available steps
"""
import os
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

LOGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "thesis", "figures")
SEEDS = [1, 8, 42, 24, 77]

LINE_STYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]


def seed_dirs(stage):
    """stage -> {seed: run_dirname} (real directories only, no .zip)."""
    if stage.startswith("nav_"):
        pat = rf"go2_navigation_(stand|walk|avoid|jump)_seed_(\d+)_"
    else:
        pat = rf"go2_petting_(stand|gestures)_seed_(\d+)_"
    out = {}
    for name in os.listdir(LOGS):
        full = os.path.join(LOGS, name)
        if not os.path.isdir(full):
            continue
        m = re.match(pat, name)
        if m and m.group(1) == stage.split("_", 1)[1]:
            out[int(m.group(2))] = full
    return out


def event_files(run_dir):
    return [os.path.join(run_dir, f) for f in os.listdir(run_dir)
            if f.startswith("events.out.tfevents")]


def stage_policy(stage):
    """Return (min_step, require_file_max_step_gt)."""
    if stage in ("nav_avoid", "nav_jump"):
        return 1049, 1049
    if stage == "nav_walk":
        return 50, None
    return 0, None   # stand (nav/petting), gestures: all


def load_seed_curve(run_dir, tag, min_step, max_gt):
    """Merge all event files of a run into one (steps, values) series (keep-first per step)."""
    merged = {}
    for p in event_files(run_dir):
        try:
            ea = EventAccumulator(p)
            ea.Reload()
        except Exception:
            continue
        if tag not in ea.Tags().get("scalars", []):
            continue
        evs = ea.Scalars(tag)
        if not evs:
            continue
        steps = [e.step for e in evs]
        if max_gt is not None and max(steps) <= max_gt:
            continue   # skip shared walk base for avoid/jump
        for e in evs:
            if e.step < min_step:
                continue
            merged.setdefault(e.step, e.value)   # keep first value per step
    if not merged:
        return None, None
    ss = sorted(merged)
    return ss, [merged[s] for s in ss]


def load_stage(stage, tag):
    dirs = seed_dirs(stage)
    min_step, max_gt = stage_policy(stage)
    curves = {}
    for seed in SEEDS:
        if seed not in dirs:
            continue
        s, v = load_seed_curve(dirs[seed], tag, min_step, max_gt)
        if s is not None and s:
            curves[seed] = (s, v)
    return curves


# figure config: output name -> (stage, tb tag)
FIGURES = [
    # Navigation stand
    ("stand_mean_episode_length_across_runs.png", "nav_stand", "Train/mean_episode_length"),
    ("stand_entropy_across_runs.png", "nav_stand", "Loss/entropy"),
    ("stand_rew_reward_across_runs.png", "nav_stand", "Episode/rew_reward"),
    ("stand_rew_height_across_runs.png", "nav_stand", "Episode/rew_height"),
    # Navigation walk
    ("walk_mean_episode_length_across_runs.png", "nav_walk", "Train/mean_episode_length"),
    ("walk_entropy_across_runs.png", "nav_walk", "Loss/entropy"),
    ("walk_rew_reward_across_runs.png", "nav_walk", "Episode/rew_reward"),
    ("walk_rew_tracking_lin_vel_across_runs.png", "nav_walk", "Episode/rew_tracking_lin_vel"),
    # Navigation avoid
    ("avoid_mean_episode_length_across_runs.png", "nav_avoid", "Train/mean_episode_length"),
    ("avoid_entropy_across_runs.png", "nav_avoid", "Loss/entropy"),
    ("avoid_rew_reward_across_runs.png", "nav_avoid", "Episode/rew_reward"),
    ("avoid_rew_target_proximity_across_runs.png", "nav_avoid", "Episode/rew_target_proximity"),
    # Navigation jump
    ("jump_mean_episode_length_across_runs.png", "nav_jump", "Train/mean_episode_length"),
    ("jump_entropy_across_runs.png", "nav_jump", "Loss/entropy"),
    ("jump_rew_reward_across_runs.png", "nav_jump", "Episode/rew_reward"),
    ("jump_rew_jump_approach_across_runs.png", "nav_jump", "Episode/rew_jump_approach"),
    ("jump_rew_jump_takeoff_across_runs.png", "nav_jump", "Episode/rew_jump_takeoff"),
    ("jump_rew_jump_flight_across_runs.png", "nav_jump", "Episode/rew_jump_flight"),
    # Petting stand
    ("stand_pet_mean_episode_length_across_runs.png", "pet_stand", "Train/mean_episode_length"),
    ("stand_pet_entropy_across_runs.png", "pet_stand", "Loss/entropy"),
    ("stand_pet_rew_reward_across_runs.png", "pet_stand", "Episode/rew_reward"),
    ("stand_pet_rew_flexible_height_across_runs.png", "pet_stand", "Episode/rew_flexible_height"),
    # Petting gestures
    ("pet_mean_episode_length_across_runs.png", "pet_gestures", "Train/mean_episode_length"),
    ("pet_entropy_across_runs.png", "pet_gestures", "Loss/entropy"),
    ("pet_rew_reward_across_runs.png", "pet_gestures", "Episode/rew_reward"),
    ("pet_rew_no_sitting_across_runs.png", "pet_gestures", "Episode/rew_no_sitting"),
    ("pet_rew_gesture_during_touch_across_runs.png", "pet_gestures", "Episode/rew_gesture_during_touch"),
    ("pet_rew_gesture_movement_across_runs.png", "pet_gestures", "Episode/rew_gesture_movement"),
]


def plot_figure(outname, stage, tag):
    curves = load_stage(stage, tag)
    tag_short = tag.split("/", 1)[-1]
    plt.figure()
    for idx, seed in enumerate(SEEDS):
        if seed not in curves:
            continue
        s, v = curves[seed]
        plt.plot(s, v, label=f"seed {seed}", linewidth=1,
                 linestyle=LINE_STYLES[idx % len(LINE_STYLES)])
    plt.title(f"{tag_short} across runs")
    plt.xlabel("Step")
    plt.ylabel(tag_short)
    plt.legend()
    plt.grid(True)
    out = os.path.join(FIGDIR, outname)
    plt.savefig(out, dpi=150)
    plt.close()
    missing = [s for s in SEEDS if s not in curves]
    print(f"{outname}: plotted {[s for s in SEEDS if s in curves]} "
          f"(missing {missing}) tag={tag}")


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    for outname, stage, tag in FIGURES:
        plot_figure(outname, stage, tag)
    print("Done.")


if __name__ == "__main__":
    main()
