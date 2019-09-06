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

brier_ary = []
att_ary = []

for fold_index in range(5):
	with open('rnn_cache_{}.pickle'.format(fold_index), 'rb') as fin:
		[att_scores_valid, gather_index_valid, ifp_valid, db_answer, db] = pickle.load(fin)


	gather_index_valid_dict = defaultdict(list)
	for row in gather_index_valid:
		gather_index_valid_dict[row[0]].append(row[1])

	for i, ifp in enumerate(ifp_valid):
		answer, is_ordered = db_answer[ifp]

		briers = []
		for f in db[ifp]:
			num_options = f[3]
			prob = f[4:9][:num_options]
			this_brier = brier(prob, answer, is_ordered)
			briers.append(this_brier)

		sent_locations = gather_index_valid_dict[i]
		score_board = []
		for t in range(len(briers)):
			score_board.append([])

		for l in sent_locations:
			atts = att_scores_valid[i, l, :]
			for z in range(l+1):
				score_board[z].append(atts[z])

		score_board_mean = [np.mean(y) for y in score_board]
		brier_ary += briers
		att_ary += score_board_mean

# each row is all requests send out from one location
# each columns represents a input forecasts waiting to be matched.

pairs = list(zip(brier_ary, att_ary))
selected = [x for x in pairs if x[1] > 0.1]
new_brier_ary, new_att_ary = zip(*selected)
#plt.scatter(new_brier_ary, new_att_ary)
sns.regplot(np.asarray(new_brier_ary, dtype=np.float32), np.asarray(new_att_ary, dtype=np.float32))
plt.ylabel('Attention Score')
plt.xlabel('Brier')
plt.show()
pdb.set_trace()
print('Leaving')
