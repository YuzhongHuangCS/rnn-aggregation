import tensorflow as tf
import pickle
import pandas as pd
import numpy as np
import pdb
from briercompute import brier as brier_compute
import os
import xgboost as xgb
import sklearn.model_selection

with open('dataset_all.pickle', 'rb') as f:
	[ifp_all, all_df] = pickle.load(f)

pdb.set_trace()
print('OK')
