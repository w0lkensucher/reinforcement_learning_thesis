import pickle
from pprint import pprint
with open('go2_locomotion/best_hyperparams_251015.pkl', 'rb') as f:
    hyperparams = pickle.load(f)

pprint(hyperparams)