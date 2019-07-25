import pandas as pd
import pdb
import dateutil.parser
import numpy as np
from collections import OrderedDict
import pickle
import sklearn.model_selection
import sklearn.decomposition
import sklearn.cluster
import scipy.stats
import random
import copy
from briercompute import brier
import torch
import torch.nn as nn

def is_ordered(opt):
	keywords = ['Less', 'Between', 'More', 'inclusive','less', 'between', 'more']
	if len(opt) == 1: #binary
		return False
	for o in opt:
		if any(x in o for x in keywords):
			return True

	return False

db_answer = {}
for filename in ('data/dump_questions_rcta.csv', 'data/dump_questions_rctb.csv'):
	df_question = pd.read_csv(filename)
	for index, row in df_question.iterrows():
		if row['is_resolved']:
			ifp_id = row['ifp_id']
			resolution = row['resolution']
			options = row.tolist()[-5:]

			clean_options = [x for x in options if type(x) == str]
			try:
				answer = options.index(resolution)
				if ifp_id in db_answer:
					pdb.set_trace()
					print(ifp_id)
				else:
					db_answer[ifp_id] = [answer, is_ordered(clean_options)]
			except ValueError as e:
				pdb.set_trace()
				print(e)

df = pd.read_csv('data/human.csv')
#df.fillna(0, inplace=True)
#pdb.set_trace()
#np.unique(df[df['ifp_id'].isin(ifp_all)]['user_id']).shape
db = OrderedDict()

for index, row in df.iterrows():
	date = row['date']
	user_id = row['user_id']
	ifp_id = row['ifp_id']

	if ifp_id not in db_answer:
		continue

	num_options = row['num_options']
	option_1 = row['option_1']
	option_2 = row['option_2']
	option_3 = row['option_3']
	option_4 = row['option_4']
	option_5 = row['option_5']

	if num_options == 1:
		num_options = 2

	if ifp_id not in db:
		db[ifp_id] = []
	else:
		if dateutil.parser.parse(date) < dateutil.parser.parse(db[ifp_id][-1][0]):
			if (dateutil.parser.parse(db[ifp_id][-1][0])-dateutil.parser.parse(date)).total_seconds() > 1:
				print('Date not in ascending order', date, db[ifp_id][-1][0])

	db[ifp_id].append([date,user_id,ifp_id,num_options,option_1,option_2,option_3,option_4,option_5])

max_steps = max([len(v) for k, v in db.items()])
n_all = len(db)

all_ifp = list(db.keys())
all_ifp_shuffle = copy.deepcopy(all_ifp)
random.seed(2019)
random.shuffle(all_ifp_shuffle)

n_train = int(n_all*0.8)
n_test = n_all - n_train

ifp_train = all_ifp_shuffle[:n_train]
ifp_test = all_ifp_shuffle[n_train:]

N_RNN_DIM = 32

### TRAIN data
input_train = np.zeros((n_train, max_steps, 5))
target_train = np.zeros((n_train, 5))
answer_train = np.zeros(n_train, dtype=int)
is_ordered_train = np.zeros(n_train, dtype=int)
seq_length_train = np.zeros(n_train, dtype=int)
gather_index_train = np.zeros((n_train, 2), dtype=int)
num_option_mask_train = np.zeros((n_train, 5), dtype=int)

for index, ifp in enumerate(ifp_train):
	forecasts = db[ifp]

	for i, forecast in enumerate(forecasts):
		input_train[index, i] = forecast[-5:]

	answer, is_ordered = db_answer[ifp]
	target_train[index, answer] = 1
	answer_train[index] = answer
	is_ordered_train[index] = is_ordered
	seq_length_train[index] = len(forecasts)

	gather_index_train[index, :] = [index, len(forecasts)-1]

	num_options = forecasts[0][3]
	num_option_mask_train[index, :num_options] = 1


input_train[np.isnan(input_train)] = 0

### TEST data
input_test = np.zeros((n_test, max_steps, 5))
target_test = np.zeros((n_test, 5))
answer_test = np.zeros(n_test, dtype=int)
is_ordered_test = np.zeros(n_test, dtype=int)
seq_length_test = np.zeros(n_test, dtype=int)
gather_index_test = np.zeros((n_test, 2), dtype=int)
num_option_mask_test = np.zeros((n_test, 5))

for index, ifp in enumerate(ifp_test):
	forecasts = db[ifp]

	for i, forecast in enumerate(forecasts):
		input_test[index, i] = forecast[-5:]

	answer, is_ordered = db_answer[ifp]
	target_test[index, answer] = 1
	answer_test[index] = answer
	is_ordered_test[index] = is_ordered
	seq_length_test[index] = len(forecasts)

	gather_index_test[index, :] = [index, len(forecasts)-1]

	num_options = forecasts[0][3]
	num_option_mask_test[index, :num_options] = 1


input_test[np.isnan(input_test)] = 0

# Network placeholder


class AggregationLayer(nn.Module):
	def __init__(self, D_in, H, D_out):
		super(AggregationLayer, self).__init__()
		self.rnn = nn.GRU(D_in, H, 1, batch_first=True)
		self.linear = nn.Linear(H, D_out, bias=True)

	def forward(self, x):
		zero_state = torch.zeros(1, n_train, N_RNN_DIM)
		state_series, _ = self.rnn(torch.from_numpy(x).float(), zero_state)
		needed_state = state_series[torch.from_numpy(gather_index_train).long().chunk(2, dim=1)].squeeze()
		prediction = self.linear(needed_state)
		softmax = nn.Softmax(dim=1)
		prob = softmax(prediction)
		return prob

model = AggregationLayer(5, N_RNN_DIM, 5)
criterion = nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
target_train = torch.from_numpy(target_train).float()
for t in range(500):
    prob = model(input_train)

    # Compute and print loss
    loss = criterion(prob, target_train)
    print(t, loss.item())

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
