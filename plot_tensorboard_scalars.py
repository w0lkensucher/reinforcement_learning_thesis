import os
import sys
import argparse
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# List of tags to plot
TAGS = [
    'Episode/rew_reward',
    'Train/mean_episode_length',
    'Loss/entropy',
    'Episode/rew_flexible_height',
]

def extract_scalars(logdir, tags):
    ea = EventAccumulator(logdir)
    ea.Reload()
    data = {}
    for tag in tags:
        if tag in ea.Tags()['scalars']:
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            data[tag] = (steps, values)
        else:
            print(f"Tag '{tag}' not found in {logdir}")
            data[tag] = ([], [])
    return data

def plot_multiple_logs(logdirs, tags, start_step=None, end_step=None):
    # Save plots in thesis/figures directory relative to the project root
    project_root = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(project_root, 'thesis', 'figures')
    os.makedirs(plots_dir, exist_ok=True)
    line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 5))]
    for tag in tags:
        plt.figure()
        # Use only the part after the slash for filename and ylabel
        tag_short = tag.split('/', 1)[-1] if '/' in tag else tag
        for idx, logdir in enumerate(logdirs):
            logdir_base = os.path.basename(logdir)
            # Extract seed number from logdir name (assumes 'seed_<number>' or 'seed<number>')
            import re
            match = re.search(r'seed[_-]?(\d+)', logdir_base)
            if match:
                label = f"seed {match.group(1)}"
            else:
                label = logdir_base
            data = extract_scalars(logdir, [tag])
            steps, values = data[tag]
            
            # Filter by step range
            if steps and (start_step is not None or end_step is not None):
                filtered = [(s, v) for s, v in zip(steps, values) 
                           if (start_step is None or s >= start_step) 
                           and (end_step is None or s <= end_step)]
                if filtered:
                    steps, values = zip(*filtered)
                    steps, values = list(steps), list(values)
                else:
                    steps, values = [], []
            
            if steps:
                style = line_styles[idx % len(line_styles)]
                plt.plot(steps, values, label=label, linewidth=1, linestyle=style)
        plt.title(f"{tag_short} across runs")
        plt.xlabel("Step")
        plt.ylabel(tag_short)
        plt.legend()
        plt.grid(True)
        # Save plot as PNG, replace slashes in tag name
        safe_tag = tag_short.replace('/', '_')
        filename = f"{safe_tag}_across_runs.png"
        plt.savefig(os.path.join(plots_dir, filename))
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot TensorBoard scalars from log directories')
    parser.add_argument('logdirs', nargs='*', help='Log directory names to plot')
    parser.add_argument('--start-step', type=int, default=None, help='Start step for data filtering')
    parser.add_argument('--end-step', type=int, default=None, help='End step for data filtering')
    args = parser.parse_args()
    
    if not args.logdirs:
        # Search for all subdirectories in the logs directory
        logs_dir = os.path.join(os.path.dirname(__file__), 'logs')
        logdirs = [os.path.join(logs_dir, d) for d in os.listdir(logs_dir)
                  if os.path.isdir(os.path.join(logs_dir, d))]
        if not logdirs:
            print(f"No log directories found in {logs_dir}")
            sys.exit(1)
        print(f"Found log directories: {logdirs}")
    else:
        logs_dir = os.path.join(os.path.dirname(__file__), 'logs')
        logdirs = []
        for arg in args.logdirs:
            # If arg is absolute or already starts with logs/, use as is
            if os.path.isabs(arg) or arg.startswith(logs_dir) or arg.startswith('logs'):
                logdirs.append(arg)
            else:
                logdirs.append(os.path.join(logs_dir, arg))
    
    if args.start_step is not None or args.end_step is not None:
        print(f"Filtering data: start_step={args.start_step}, end_step={args.end_step}")
    
    plot_multiple_logs(logdirs, TAGS, args.start_step, args.end_step)

if __name__ == "__main__":
    main()
