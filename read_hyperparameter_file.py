import pickle
from pprint import pprint
with open('logs/go2_petting_fall_calm_gone_20251218/env_cfg_final.pkl', 'rb') as f:
    hyperparams = pickle.load(f)

pprint(hyperparams)