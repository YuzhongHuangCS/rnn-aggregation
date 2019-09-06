import tensorflow as tf
import pickle
import pandas as pd
import numpy as np
import pdb
from briercompute import brier as brier_compute
import os
import xgboost as xgb

# uncomment to force CPU training
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# optimize CPU performance
#os.environ['KMP_BLOCKTIME'] = '0'
#os.environ['KMP_AFFINITY'] = 'granularity=fine,verbose,compact,1,0'

with open('dataset_0.pickle', 'rb') as f:
	train_df, test_df = pickle.load(f)

'''
drop_list = ['entropy_b', 'entropy_c',
	   'entropy_d', 'entropy_human', 'entropy_sage', 'entropy_te',
	   'n_forecasts', 'n_forecasts_b', 'n_forecasts_c', 'n_forecasts_d',
	   'n_forecasts_sage', 'n_forecasts_te', 'ordinal', 'p_updates', 'stage',
	   'variance_b', 'variance_c', 'variance_d', 'variance_human',
	   'variance_sage', 'variance_te', 'x_acf1', 'x_acf10', 'diff1_acf1',
	   'diff1_acf10', 'diff2_acf1', 'diff2_acf10', 'seas_acf1', 'ARCH.LM',
	   'crossing_points', 'entropy', 'flat_spots', 'arch_acf', 'garch_acf',
	   'arch_r2', 'garch_r2', 'alpha', 'beta', 'hurst', 'lumpiness',
	   'nonlinearity', 'x_pacf5', 'diff1x_pacf5', 'diff2x_pacf5', 'seas_pacf',
	   'nperiods', 'seasonal_period', 'trend', 'spike', 'linearity',
	   'curvature', 'e_acf1', 'e_acf10', 'seasonal_strength', 'peak', 'trough',
	   'stability', 'hw_alpha', 'hw_beta', 'hw_gamma', 'unitroot_kpss',
	   'unitroot_pp', 'series_length']
'''
drop_list = [
	   'n_forecasts', 'n_forecasts_b', 'n_forecasts_c', 'n_forecasts_d',
	   'n_forecasts_sage', 'n_forecasts_te', 'ordinal', 'p_updates', 'x_acf1', 'x_acf10', 'diff1_acf1',
	   'diff1_acf10', 'diff2_acf1', 'diff2_acf10', 'seas_acf1', 'ARCH.LM',
	   'crossing_points', 'entropy', 'flat_spots', 'arch_acf', 'garch_acf',
	   'arch_r2', 'garch_r2', 'alpha', 'beta', 'hurst', 'lumpiness',
	   'nonlinearity', 'x_pacf5', 'diff1x_pacf5', 'diff2x_pacf5', 'seas_pacf',
	   'nperiods', 'seasonal_period', 'trend', 'spike', 'linearity',
	   'curvature', 'e_acf1', 'e_acf10', 'seasonal_strength', 'peak', 'trough',
	   'hw_alpha', 'hw_beta', 'hw_gamma', 'unitroot_kpss',
	   'unitroot_pp']

drop_model_list = [
	'arw_1', 'arw_2', 'arw_3', 'arw_4', 'arw_5',
	'dsholt_1', 'dsholt_2', 'dsholt_3', 'dsholt_4', 'dsholt_5',
	'dsholtd_1', 'dsholtd_2', 'dsholtd_3', 'dsholtd_4', 'dsholtd_5',
	'dsrw_1', 'dsrw_2', 'dsrw_3', 'dsrw_4', 'dsrw_5',
	'dsses_1', 'dsses_2', 'dsses_3', 'dsses_4', 'dsses_5',
	'ets_1', 'ets_2', 'ets_3', 'ets_4', 'ets_5',
	'grw_1', 'grw_2', 'grw_3', 'grw_4', 'grw_5',
	'm4c_1', 'm4c_2', 'm4c_3', 'm4c_4', 'm4c_5',
	'mean_1', 'mean_2', 'mean_3', 'mean_4', 'mean_5',
	'nnetar_1', 'nnetar_2', 'nnetar_3', 'nnetar_4', 'nnetar_5',
	'rnn_1', 'rnn_2', 'rnn_3', 'rnn_4', 'rnn_5',
	'rw_1', 'rw_2', 'rw_3', 'rw_4', 'rw_5',
	'rwd_1', 'rwd_2', 'rwd_3', 'rwd_4', 'rwd_5',
	'rws_1', 'rws_2', 'rws_3', 'rws_4', 'rws_5',
	#'sim_1', 'sim_2', 'sim_3', 'sim_4', 'sim_5',
	'sltmar_1', 'sltmar_2', 'sltmar_3', 'sltmar_4', 'sltmar_5',
	'tbats_1', 'tbats_2', 'tbats_3', 'tbats_4', 'tbats_5',
	'theta_1', 'theta_2', 'theta_3', 'theta_4', 'theta_5'
]

train_X = train_df.drop(columns=['ifp_id', 'date', 'answer', 'ordered']).drop(columns=drop_list)
print(train_X.columns.tolist())
train_X = train_X.values
train_Y = train_df['answer'].values
train_Z = (train_df['ordered'].values.astype(int) - 0.5)*2

train_count_db = {}
for ifp in set(train_df['ifp_id']):
	index = train_df.index[train_df['ifp_id'] == ifp]
	train_count_db[ifp] = 1.0/len(index)

train_W = []
for index, row in train_df.iterrows():
	train_W.append(train_count_db[row['ifp_id']])

train_W = np.asarray(train_W)
train_W /= len(set(train_df['ifp_id']))
train_W *= len(train_df)
train_W_modulate = [i*j for i, j in zip(train_W, train_Z)]

test_X = test_df.drop(columns=['ifp_id', 'date', 'answer', 'ordered']).drop(columns=drop_list).values
test_Y = test_df['answer'].values
test_Z = (test_df['ordered'].values.astype(int) - 0.5)*2

test_count_db = {}
for ifp in set(test_df['ifp_id']):
	index = test_df.index[test_df['ifp_id'] == ifp]
	test_count_db[ifp] = 1.0/len(index)

test_W = []
for index, row in test_df.iterrows():
	test_W.append(test_count_db[row['ifp_id']])

test_W = np.asarray(test_W)
test_W /= len(set(test_df['ifp_id']))
test_W *= len(test_df)
test_W_modulate = [i*j for i, j in zip(test_W, test_Z)]

n_features = train_X.shape[1]
print('n_features', n_features)

def brier_gradient(optionprobs, correct, ordered=False):
	"""
		This computes the brier score for a user on
		a *single* day (daily averages are handled in a different function).

		optionprobs: a LIST of answer probabilities. Should sum to 1.
			In other words, it assumes that *all* possibilites are accounted for.
			e.g. [0.64, 0.36] for a binary question.
			In the case of an ordered multinomial, it assumes that the ordering is correct.

		correct: the index of the correct answer. Zero-based.

		ordered: a boolean flag determining whether or not the answer is *ordered* multinomial.
			If so, the formula is substantially different.
	"""

	# make a one-hot vector for the true outcome.
	true_outcomes = [0] * len(optionprobs)
	true_outcomes[correct] = 1
	assert len(optionprobs) == 5

	brier = np.asarray([0.0] * len(optionprobs))
	if ordered:
		a, b, c, d, e = optionprobs
		A, B, C, D, E = true_outcomes

		grad_a = 0.5*(4*(a-A) + 3*(b-B) + 2*(c-C) + 1*(d-D) + 0*(e-E))
		grad_b = 0.5*(3*(a-A) + 4*(b-B) + 3*(c-C) + 2*(d-D) + 1*(e-E))
		grad_c = 0.5*(2*(a-A) + 3*(b-B) + 4*(c-C) + 3*(d-D) + 2*(e-E))
		grad_d = 0.5*(1*(a-A) + 2*(b-B) + 3*(c-C) + 4*(d-D) + 3*(e-E))
		grad_e = 0.5*(0*(a-A) + 1*(b-B) + 2*(c-C) + 3*(d-D) + 4*(e-E))

		brier = [grad_a, grad_b, grad_c, grad_d, grad_e]
	else:
		brier = [2*(i - j) for i, j in zip(optionprobs, true_outcomes)]

	return np.asarray(brier)

def logregobj(preds, dtrain):
	labels = dtrain.get_label()
	#pdb.set_trace()
	#preds_exp = np.exp(preds)
	#preds_exp_softmax = np.asarray([i / j for i, j in zip(preds_exp, np.sum(preds_exp, axis=1))])

	grad = np.asarray([brier_gradient(value, int(labels[index]), train_Z[index])*train_W[index] for index, value in enumerate(preds)])
	#hess = np.ones(grad.shape) * 2
	hess = np.asarray([[v, v, v, v, v] for i, v in enumerate(train_W)]) * 2
	return grad.reshape(-1), hess.reshape(-1)

def evalerror(preds, dtrain):
	labels = dtrain.get_label()
	weights = dtrain.get_weight()

	print('eval', labels.shape, sum([abs(x-1)>0.0001 for x in np.sum(preds, axis=1)]))
	#preds_exp = np.exp(preds)
	#preds_exp_softmax = np.asarray([i / j for i, j in zip(preds_exp, np.sum(preds_exp, axis=1))])

	if len(labels) == len(train_Y):
		briers = [brier_compute(value, int(labels[index]), train_Z[index])*train_W[index] for index, value in enumerate(preds)]
	else:
		assert len(labels) == len(test_Y)
		briers = [brier_compute(value, int(labels[index]), test_Z[index])*test_W[index] for index, value in enumerate(preds)]

	return 'brier', np.mean(briers)

dtrain = xgb.DMatrix(train_X, train_Y, weight=train_W_modulate)
dtest = xgb.DMatrix(test_X, test_Y, weight=test_W_modulate)
param = {'eta': 1, 'gamma': 0.1, 'max_depth': 3, 'objective': 'multi:softprob', 'num_class': 5, 'verbosity': 3, 'lambda': 1, 'alpha': 1, 'disable_default_eval_metric': 1}
watchlist = [(dtrain, 'train'), (dtest, 'eval')]
num_round = 100
num_stopping_rounds = 10

bst = xgb.train(param, dtrain, num_round, watchlist, obj=logregobj, feval=evalerror, early_stopping_rounds=num_stopping_rounds)
print(bst.best_iteration, bst.best_ntree_limit)
test_Y_prob = bst.predict(dtest, ntree_limit=bst.best_ntree_limit)

test_Y_brier = []
for i, v in enumerate(test_Y_prob):
	row = test_df.iloc[i]
	ifp_id = row['ifp_id']
	answer = row['answer']
	ordered = row['ordered']

	human_brier = brier_compute([row['h_1'], row['h_2'], row['h_3'], row['h_4'], row['h_5']], answer, ordered)
	human1_brier = brier_compute([row['h1_1'], row['h1_2'], row['h1_3'], row['h1_4'], row['h1_5']], answer, ordered)
	human2_brier = brier_compute([row['h2_1'], row['h2_2'], row['h2_3'], row['h2_4'], row['h2_5']], answer, ordered)
	human3_brier = brier_compute([row['h3_1'], row['h3_2'], row['h3_3'], row['h3_4'], row['h3_5']], answer, ordered)
	human4_brier = brier_compute([row['h4_1'], row['h4_2'], row['h4_3'], row['h4_4'], row['h4_5']], answer, ordered)
	arima_brier = brier_compute([row['arima_1'], row['arima_2'], row['arima_3'], row['arima_4'], row['arima_5']], answer, ordered)
	m4_brier = brier_compute([row['m4_1'], row['m4_2'], row['m4_3'], row['m4_4'], row['m4_5']], answer, ordered)
	arw_brier = brier_compute([row['arw_1'], row['arw_2'], row['arw_3'], row['arw_4'], row['arw_5']], answer, ordered)
	dsholt_brier = brier_compute([row['dsholt_1'], row['dsholt_2'], row['dsholt_3'], row['dsholt_4'], row['dsholt_5']], answer, ordered)
	dsholtd_brier = brier_compute([row['dsholtd_1'], row['dsholtd_2'], row['dsholtd_3'], row['dsholtd_4'], row['dsholtd_5']], answer, ordered)
	dsrw_brier = brier_compute([row['dsrw_1'], row['dsrw_2'], row['dsrw_3'], row['dsrw_4'], row['dsrw_5']], answer, ordered)
	dsses_brier = brier_compute([row['dsses_1'], row['dsses_2'], row['dsses_3'], row['dsses_4'], row['dsses_5']], answer, ordered)
	ets_brier = brier_compute([row['ets_1'], row['ets_2'], row['ets_3'], row['ets_4'], row['ets_5']], answer, ordered)
	grw_brier = brier_compute([row['grw_1'], row['grw_2'], row['grw_3'], row['grw_4'], row['grw_5']], answer, ordered)
	m4c_brier = brier_compute([row['m4c_1'], row['m4c_2'], row['m4c_3'], row['m4c_4'], row['m4c_5']], answer, ordered)
	mean_brier = brier_compute([row['mean_1'], row['mean_2'], row['mean_3'], row['mean_4'], row['mean_5']], answer, ordered)
	nnetar_brier = brier_compute([row['nnetar_1'], row['nnetar_2'], row['nnetar_3'], row['nnetar_4'], row['nnetar_5']], answer, ordered)
	rnn_brier = brier_compute([row['rnn_1'], row['rnn_2'], row['rnn_3'], row['rnn_4'], row['rnn_5']], answer, ordered)
	rw_brier = brier_compute([row['rw_1'], row['rw_2'], row['rw_3'], row['rw_4'], row['rw_5']], answer, ordered)
	rwd_brier = brier_compute([row['rwd_1'], row['rwd_2'], row['rwd_3'], row['rwd_4'], row['rwd_5']], answer, ordered)
	rws_brier = brier_compute([row['rws_1'], row['rws_2'], row['rws_3'], row['rws_4'], row['rws_5']], answer, ordered)
	#sim_brier = brier_compute([row['sim_1'], row['sim_2'], row['sim_3'], row['sim_4'], row['sim_5']], answer, ordered)
	sltmar_brier = brier_compute([row['sltmar_1'], row['sltmar_2'], row['sltmar_3'], row['sltmar_4'], row['sltmar_5']], answer, ordered)
	tbats_brier = brier_compute([row['tbats_1'], row['tbats_2'], row['tbats_3'], row['tbats_4'], row['tbats_5']], answer, ordered)
	theta_brier = brier_compute([row['theta_1'], row['theta_2'], row['theta_3'], row['theta_4'], row['theta_5']], answer, ordered)
	predict_brier = brier_compute(v, answer, ordered)
	test_Y_brier.append([human_brier, human1_brier, human2_brier, human3_brier, human4_brier, arima_brier, m4_brier, arw_brier, dsholt_brier, dsholtd_brier, dsrw_brier, dsses_brier, ets_brier, grw_brier, m4c_brier, mean_brier, nnetar_brier, rnn_brier, rw_brier, rwd_brier, rws_brier, sltmar_brier, tbats_brier, theta_brier, predict_brier])
	#test_Y_brier.append([human_brier, arima_brier, m4_brier, arw_brier, dsholt_brier, dsholtd_brier, dsrw_brier, dsses_brier, ets_brier, grw_brier, m4c_brier, mean_brier, nnetar_brier, rnn_brier, rw_brier, rwd_brier, rws_brier, sltmar_brier, tbats_brier, theta_brier, predict_brier])
	#test_Y_brier.append([human_brier, human1_brier, human2_brier, human3_brier, human4_brier, arima_brier, m4_brier, predict_brier])
test_Y_brier = np.asarray(test_Y_brier)

db_brier = {}

for ifp in set(test_df['ifp_id']):
	index = test_df.index[test_df['ifp_id'] == ifp]
	scores = np.mean(test_Y_brier[index], axis=0).tolist()
	two_ideal = np.mean(np.min([test_Y_brier[index][:, 0], test_Y_brier[index][:, 5]], axis=0))
	ideal = np.mean(np.min(test_Y_brier[index][:, :-1], axis=1))
	scores.insert(-1, two_ideal)
	scores.insert(-1, ideal)
	db_brier[ifp] = np.asarray(scores)

briers = np.asarray(list(db_brier.values()))
briers_mean = np.mean(briers, axis=0)
print('human', 'arima', 'two_ideal', 'ideal', 'predict')
print(briers_mean[0], briers_mean[5], briers_mean[-3], briers_mean[-2], briers_mean[-1])
