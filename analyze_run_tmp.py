from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

path = 'logs/go2_navigation_goal_jump_walking_advanced_curr_20260311'
ea = EventAccumulator(path)
ea.Reload()

scalars = ea.Tags()['scalars']

def summarize(tag, window=50):
    evs = ea.Scalars(tag)
    vals = [e.value for e in evs]
    steps = [e.step for e in evs]
    n = len(vals)
    if n == 0:
        return
    # Print every N steps to show whole trajectory
    stride = max(1, n // 30)
    print(f'\n  [{tag}] ({n} iters, last step={steps[-1]})')
    for i in range(0, n, stride):
        print(f'    step={steps[i]:>5}, val={vals[i]:.4f}')
    print(f'    step={steps[-1]:>5}, val={vals[-1]:.4f}  <-- CURRENT')
    # Recent trend: last 200 vs prev 200
    if n >= 100:
        recent = sum(vals[-100:]) / 100
        older = sum(vals[max(0,n-200):max(1,n-100)]) / min(100, n-100)
        trend = 'IMPROVING' if recent > older else 'DECLINING'
        print(f'    Avg last 100: {recent:.4f} vs prev 100: {older:.4f} -> {trend}')

print('=== advanced_curr: FULL TRAJECTORY ===')
summarize('Train/mean_reward')
summarize('Train/mean_episode_length')
summarize('Loss/entropy')
summarize('Loss/value_function')
summarize('Loss/surrogate')
summarize('Loss/learning_rate')
summarize('Policy/mean_noise_std')

print('\n\n=== REWARD COMPONENTS (last 5 values) ===')
rew_tags = [t for t in scalars if t.startswith('Episode/rew_') and t != 'Episode/rew_reward']
for tag in rew_tags:
    evs = ea.Scalars(tag)
    vals = [e.value for e in evs]
    steps = [e.step for e in evs]
    n = len(vals)
    last5 = list(zip(steps[-5:], [round(v,4) for v in vals[-5:]]))
    first_nonzero = next((vals[i] for i in range(n) if vals[i] != 0.0), 0.0)
    short = tag[len('Episode/rew_'):]
    trend = 'UP' if n>50 and vals[-1] > vals[max(0,n-50)] else 'DOWN'
    print(f'  {short:30s}: last={vals[-1]:.4f}, [{trend}] {last5}')
