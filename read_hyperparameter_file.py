import pickle
from pprint import pprint
with open('logs/hyperopt/go2_obstacles_20251205.pkl', 'rb') as f:
    hyperparams = pickle.load(f)

pprint(hyperparams)