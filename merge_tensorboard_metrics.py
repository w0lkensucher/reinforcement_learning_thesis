import os
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import numpy as np

# Paths to your event files (update these to your actual files)
old_event_path = 'logs/go2_petting_standing_works_20260108 - resume_from_this'  # before checkpoint
new_event_path = 'logs/go2_petting_refining_scaling_20260108'    # after resume

def load_scalars(event_dir):
    ea = event_accumulator.EventAccumulator(event_dir)
    ea.Reload()
    tags = ea.Tags()['scalars']
    scalars = {}
    for tag in tags:
        events = ea.Scalars(tag)
        steps = np.array([e.step for e in events])
        values = np.array([e.value for e in events])
        scalars[tag] = (steps, values)
    return scalars

def merge_and_plot(old_dir, new_dir):
    merge_step = 150  # Set your merge point here
    output_dir = os.path.join(new_dir, 'merged_outputs')  # Directory to save CSVs and PNGs
    os.makedirs(output_dir, exist_ok=True)
    old_scalars = load_scalars(old_dir)
    new_scalars = load_scalars(new_dir)
    all_tags = set(old_scalars.keys()) | set(new_scalars.keys())
    for tag in all_tags:
        plt.figure()
        # Old data (keep only points <= merge_step)
        if tag in old_scalars:
            steps_old, vals_old = old_scalars[tag]
            mask_old = steps_old <= merge_step
            steps_old, vals_old = steps_old[mask_old], vals_old[mask_old]
            plt.plot(steps_old, vals_old, label='Before Resume')
            last_step = steps_old[-1] if len(steps_old) > 0 else 0
        else:
            steps_old, vals_old = np.array([]), np.array([])
            last_step = 0
        # New data (keep only points > merge_step)
        if tag in new_scalars:
            steps_new, vals_new = new_scalars[tag]
            mask_new = steps_new > merge_step
            steps_new, vals_new = steps_new[mask_new], vals_new[mask_new]
            # Shift new steps to continue from old
            if len(steps_new) > 0 and steps_new[0] <= last_step:
                steps_new = steps_new + last_step + 1 - steps_new[0]
            plt.plot(steps_new, vals_new, label='After Resume')
        else:
            steps_new, vals_new = np.array([]), np.array([])

        # Connect the last old point to the first new point if both exist
        if len(steps_old) > 0 and len(steps_new) > 0:
            plt.plot([steps_old[-1], steps_new[0]], [vals_old[-1], vals_new[0]], 'k--', alpha=0.5, label='Merge Connection')
        # Export to CSV
        import csv
        csv_filename = os.path.join(output_dir, f'merged_{tag.replace("/", "_")}.csv')
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['step', 'value', 'source'])
            for s, v in zip(steps_old, vals_old):
                writer.writerow([s, v, 'before'])
            for s, v in zip(steps_new, vals_new):
                writer.writerow([s, v, 'after'])
        # Plot
        plt.title(tag)
        plt.xlabel('Step')
        plt.ylabel(tag)
        plt.legend()
        plt.tight_layout()
        png_filename = os.path.join(output_dir, f'merged_{tag.replace("/", "_")}.png')
        plt.savefig(png_filename)
        plt.close()
    print(f'Plots and CSVs saved in {output_dir} as merged_<tag>.png and merged_<tag>.csv for each metric, merged at step {merge_step}.')

if __name__ == '__main__':
    merge_and_plot(old_event_path, new_event_path)
