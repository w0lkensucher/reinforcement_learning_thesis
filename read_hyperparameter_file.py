import pickle
from pprint import pprint
with open('C:\\Users\\Anouv\\Documents\\reinforcement_learning_thesis\\logs\\go2_navigation_walking_less_standing_20260219\\cfgs.pkl', 'rb') as f:
    hyperparams = pickle.load(f)

pprint(hyperparams)