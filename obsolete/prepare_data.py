import dateparser
import pdb
import pickle
import os
import pandas as pd
import sklearn
import sklearn.linear_model
import sklearn.ensemble
import sklearn.svm
import sklearn.neural_network
import numpy as np
from briercompute import brier as brier_compute
import scipy.optimize
import matplotlib.pyplot as plt

ifp_list_filename = 'ifp_list.pickle'
if os.path.isfile(ifp_list_filename):
	with open(ifp_list_filename, 'rb') as f:
		ifp_train, ifp_test = pickle.load(f)
else:
	ifp_db = {}
	with open('slot_performance.csv') as fin:
		fin.readline()
		for line in fin:
			parts = line.split(',')
			date = dateparser.parse(parts[0])
			ifp = parts[1]
			if ifp in ifp_db:
				prev_date = ifp_db[ifp]
				if date < prev_date:
					ifp_db[ifp] = date
			else:
				ifp_db[ifp] = date

	ifp_list = [(x[1], x[0]) for x in ifp_db.items()]
	ifp_list.sort()

	n_train = int(len(ifp_list) * 0.8)
	n_test = len(ifp_list)-n_train
	ifp_train = set([int(x[1]) for x in ifp_list[:n_train]])
	ifp_test = set([int(x[1]) for x in ifp_list[n_train:]])

	with open(ifp_list_filename, 'wb') as f:
		pickle.dump([ifp_train, ifp_test], f, pickle.HIGHEST_PROTOCOL)
