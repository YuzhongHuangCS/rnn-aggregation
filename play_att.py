import os

# uncomment to force CPU training
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# optimize CPU performance
os.environ['KMP_BLOCKTIME'] = '0'
os.environ['KMP_AFFINITY'] = 'granularity=fine,verbose,compact,1,0'

import pandas as pd
import pdb
import sys
import dateutil.parser
import numpy as np
from collections import OrderedDict, Counter, defaultdict
import pickle
import sklearn.model_selection
import random
import copy
from briercompute import brier
from datetime import datetime, timedelta
import math
from nltk.tokenize import word_tokenize
from utils import is_ordered, initialize_embedding, embedding_lookup
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

brier_ary = []
att_ary = []

for fold_index in range(5):
	with open('att_cache_{}.pickle'.format(fold_index), 'rb') as fin:
		[att_scores_valid1, att_scores_valid2, gather_index_valid, ifp_valid, db_answer, db] = pickle.load(fin)


	gather_index_valid_dict = defaultdict(list)
	for row in gather_index_valid:
		gather_index_valid_dict[row[0]].append(row[1])

	for i, ifp in enumerate(ifp_valid):
		answer, is_ordered = db_answer[ifp]

		day_brier = OrderedDict()
		briers = []

		for f in db[ifp]:
			date = datetime.strftime(f[0], "%Y-%m-%d")
			num_options = f[3]
			prob = f[4:9][:num_options]
			this_brier = brier(prob, answer, is_ordered)
			briers.append(this_brier)

		'''
			if date not in day_brier:
				day_brier[date] = []

			day_brier[date].append(this_brier)


		for date, today_brier in day_brier.items():
			#scaler = StandardScaler()
			#today_brier = scaler.fit_transform(np.asarray(today_brier).reshape(-1, 1)).squeeze().tolist()
			if isinstance(today_brier, float):
				today_brier = [today_brier]
			briers += today_brier
		'''
		sent_locations = gather_index_valid_dict[i]
		score_board = []
		for t in range(len(db[ifp])):
			score_board.append([])

		for l in sent_locations:
			atts1 = att_scores_valid1[i, l, :]
			atts2 = att_scores_valid2[i, l, :]
			for z in range(l+1):
				score_board[z].append(atts2[z])

		score_board_mean = [np.mean(y) for y in score_board]
		brier_ary += briers
		att_ary += score_board_mean

# each row is all requests send out from one location
# each columns represents a input forecasts waiting to be matched.

print(len(brier_ary), len(att_ary))
pairs = list(zip(brier_ary, att_ary))
selected = [x for x in pairs]
new_brier_ary, new_att_ary = zip(*selected)
from sklearn.neighbors import KernelDensity

def kde2D(x, y, bandwidth, xbins=100j, ybins=100j, **kwargs):
    """Build 2D kernel density estimate (KDE)."""

    # create grid of sample locations (default: 100x100)
    xx, yy = np.mgrid[x.min():x.max():xbins,
                      y.min():y.max():ybins]

    xy_sample = np.vstack([yy.ravel(), xx.ravel()]).T
    xy_train  = np.vstack([y, x]).T

    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(xy_train)

    # score_samples() returns the log-likelihood of the samples
    z = np.exp(kde_skl.score_samples(xy_sample))
    return xx, yy, np.reshape(z, xx.shape)

xx, yy, zz = kde2D(new_brier_ary, y, 1.0)

#plt.scatter(new_brier_ary, new_att_ary)

#sns.regplot(np.asarray(new_brier_ary, dtype=np.float32), np.asarray(new_att_ary, dtype=np.float32))
plt.ylabel('Attention Score')
plt.xlabel('Brier')
plt.show()
pdb.set_trace()
print('Leaving')
