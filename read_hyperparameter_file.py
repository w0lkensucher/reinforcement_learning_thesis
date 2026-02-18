import pickle
from pprint import pprint
with open('C:\\Users\\Anouv\\Documents\\reinforcement_learning_thesis\\logs\\go2_navigation_walking_20260117\\cfgs.pkl', 'rb') as f:
    hyperparams = pickle.load(f)

pprint(hyperparams)