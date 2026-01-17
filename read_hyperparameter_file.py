import pickle
from pprint import pprint
with open('C:\\Users\\Anouv\\Documents\\reinforcement_learning_thesis\\logs\\go2_petting_standing_works_20260108 - resume_from_this\\cfgs.pkl', 'rb') as f:
    hyperparams = pickle.load(f)

pprint(hyperparams)