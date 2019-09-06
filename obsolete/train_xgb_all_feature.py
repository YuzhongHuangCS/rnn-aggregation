import tensorflow as tf
import pickle
import pandas as pd
import numpy as np
import pdb
from briercompute import brier as brier_compute
import os
import xgboost as xgb
import sklearn.model_selection
import matplotlib.pyplot as plt
import shap

with open('dataset_all.pickle', 'rb') as f:
	[ifp_all, all_df] = pickle.load(f)

task_name = 'all_feature'
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
		#'arima_1', 'arima_2', 'arima_3', 'arima_4', 'arima_5',
		#'m4_1', 'm4_2', 'm4_3', 'm4_4', 'm4_5', 'Health/Disease', 'Macroeconomics/Finance', 'Natural Sciences/Climate', 'Other', 'Politics/Intl Relations', 'Technology', 'entropy_b', 'entropy_c', 'entropy_d', 'entropy_human', 'entropy_sage', 'entropy_te', 'stage', 'variance_b', 'variance_c', 'variance_d', 'variance_human', 'variance_sage', 'variance_te', 'stability', 'series_length', 'skew', 'ratio',
		#'h1_1', 'h1_2', 'h1_3', 'h1_4', 'h1_5', 'h2_1', 'h2_2', 'h2_3', 'h2_4', 'h2_5',
		'h1_1', 'h1_2', 'h1_3', 'h1_4', 'h1_5', 'h2_1', 'h2_2', 'h2_3', 'h2_4', 'h2_5', 'h3_1', 'h3_2', 'h3_3', 'h3_4', 'h3_5', 'h4_1', 'h4_2', 'h4_3', 'h4_4', 'h4_5',
	   #'n_forecasts', 'n_forecasts_b', 'n_forecasts_c', 'n_forecasts_d',
	   #'n_forecasts_sage', 'n_forecasts_te', 'ordinal', 'p_updates', 'x_acf1', 'x_acf10', 'diff1_acf1',
	   #'diff1_acf10', 'diff2_acf1', 'diff2_acf10', 'seas_acf1', 'ARCH.LM',
	   #'crossing_points', 'entropy', 'flat_spots', 'arch_acf', 'garch_acf',
	   #'arch_r2', 'garch_r2', 'alpha', 'beta', 'hurst', 'lumpiness',
	   #'nonlinearity', 'x_pacf5', 'diff1x_pacf5', 'diff2x_pacf5', 'seas_pacf',
	   #'nperiods', 'seasonal_period', 'trend', 'spike', 'linearity',
	   #'curvature', 'e_acf1', 'e_acf10', 'seasonal_strength', 'peak', 'trough',
	   #'hw_alpha', 'hw_beta', 'hw_gamma', 'unitroot_kpss',
	   #'unitroot_pp']
	   ]

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


def brier_gradient(optionprobs, correct, ordered=False):
	true_outcomes = [0] * len(optionprobs)
	true_outcomes[correct] = 1
	'''
	random_number = np.random.uniform(0, 1)
	if correct >= 1:
		if correct < (len(optionprobs)-1):
			true_outcomes[correct] = 0.9
			true_outcomes[correct-1] = 0.05
			true_outcomes[correct+1] = 0.05
		else:
			true_outcomes[correct] = 0.95
			true_outcomes[correct-1] = 0.05
	else:
		if correct < (len(optionprobs)-1):
			true_outcomes[correct] = 0.95
			true_outcomes[correct+1] = 0.05
		else:
			print('It should never happen')
			pdb.set_trace()
	'''
	'''
	if random_number < 0.5:
		true_outcomes[correct] = 1
	elif random_number < 0.8:
		if correct >= 1:
			if correct < (len(optionprobs)-1):
				true_outcomes[correct] = 0.9
				true_outcomes[correct-1] = 0.05
				true_outcomes[correct+1] = 0.05
			else:
				true_outcomes[correct] = 0.95
				true_outcomes[correct-1] = 0.05
		else:
			if correct < (len(optionprobs)-1):
				true_outcomes[correct] = 0.95
				true_outcomes[correct+1] = 0.05
			else:
				print('It should never happen')
				pdb.set_trace()
	else:
		if correct >= 1:
			if correct < (len(optionprobs)-1):
				true_outcomes[correct] = 0.8
				true_outcomes[correct-1] = 0.1
				true_outcomes[correct+1] = 0.1
			else:
				true_outcomes[correct] = 0.9
				true_outcomes[correct-1] = 0.1
		else:
			if correct < (len(optionprobs)-1):
				true_outcomes[correct] = 0.9
				true_outcomes[correct+1] = 0.1
			else:
				print('It should never happen')
				pdb.set_trace()
	'''
	if ordered and len(optionprobs) >= 3:
		if len(optionprobs) == 5:
			a, b, c, d, e = optionprobs
			A, B, C, D, E = true_outcomes

			grad_a = 2/4*(4*(a-A) + 3*(b-B) + 2*(c-C) + 1*(d-D) + 0*(e-E))
			grad_b = 2/4*(3*(a-A) + 4*(b-B) + 3*(c-C) + 2*(d-D) + 1*(e-E))
			grad_c = 2/4*(2*(a-A) + 3*(b-B) + 4*(c-C) + 3*(d-D) + 2*(e-E))
			grad_d = 2/4*(1*(a-A) + 2*(b-B) + 3*(c-C) + 4*(d-D) + 3*(e-E))
			grad_e = 2/4*(0*(a-A) + 1*(b-B) + 2*(c-C) + 3*(d-D) + 4*(e-E))

			brier = [grad_a, grad_b, grad_c, grad_d, grad_e]
		elif len(optionprobs) == 4:
			a, b, c, d = optionprobs
			A, B, C, D = true_outcomes

			grad_a = 2/3*(3*(a-A) + 2*(b-B) + 1*(c-C) + 0*(d-D))
			grad_b = 2/3*(2*(a-A) + 3*(b-B) + 2*(c-C) + 1*(d-D))
			grad_c = 2/3*(1*(a-A) + 2*(b-B) + 3*(c-C) + 2*(d-D))
			grad_d = 2/3*(0*(a-A) + 1*(b-B) + 2*(c-C) + 3*(d-D))
			brier = [grad_a, grad_b, grad_c, grad_d]
		else:
			assert len(optionprobs) == 3
			a, b, c = optionprobs
			A, B, C = true_outcomes

			grad_a = 2/2*(2*(a-A) + 1*(b-B) + 0*(c-C))
			grad_b = 2/2*(1*(a-A) + 2*(b-B) + 1*(c-C))
			grad_c = 2/2*(0*(a-A) + 1*(b-B) + 2*(c-C))
			brier = [grad_a, grad_b, grad_c]
	else:
		brier = [2*(i - j) for i, j in zip(optionprobs, true_outcomes)]

	return np.pad(brier, (0, 5-len(brier)), 'constant', constant_values=0)

def logregobj(preds, dtrain):
	labels = dtrain.get_label()
	weights = dtrain.get_weight()

	num_options = np.floor(labels)
	labels = (labels * 10) % 10

	grad = []
	hess = []
	for index, value in enumerate(preds):
		num_option = int(num_options[index])
		prob = value[:num_option]
		prob /= sum(prob)

		this_grad = brier_gradient(prob, int(labels[index]), weights[index] > 0)*np.abs(weights[index])
		grad.append(this_grad)

	for index, value in enumerate(weights):
		this_hess = [np.abs(value)] * 5
		hess.append(this_hess)

	grad = np.asarray(grad)
	hess = np.asarray(hess) * 2
	return grad.reshape(-1), hess.reshape(-1)

def evalerror(preds, dtrain):
	labels = dtrain.get_label()
	weights = dtrain.get_weight()

	num_options = np.floor(labels)
	labels = (labels * 10) % 10


	briers = []
	for index, value in enumerate(preds):
		num_option = int(num_options[index])
		prob = value[:num_option]
		prob /= sum(prob)

		this_brier = brier_compute(prob, int(labels[index]), weights[index] > 0)*np.abs(weights[index])
		briers.append(this_brier)

	return 'brier', np.mean(briers)

kf = sklearn.model_selection.KFold(shuffle=True, n_splits=5, random_state=2019)
counter = 0
splits = []
for train_index, test_index in kf.split(ifp_all):
	ifp_train = set(ifp_all[train_index])
	ifp_test = set(ifp_all[test_index])
	train_df = all_df[all_df['ifp_id'].isin(ifp_train)].reset_index().drop(columns=['index'])
	test_df = all_df[all_df['ifp_id'].isin(ifp_test)].reset_index().drop(columns=['index'])

	with open('dataset_{}.pickle'.format(counter), 'wb') as f:
		pickle.dump([train_df, test_df], f, pickle.HIGHEST_PROTOCOL)

	train_X = train_df.drop(columns=['ifp_id', 'date', 'answer', 'ordered', 'num_options']).drop(columns=drop_list)
	feature_names = train_X.columns.tolist()
	print(feature_names)
	train_X = train_X.values
	train_Y = train_df['answer'].values
	train_Y_modulate = [i+j/10 for i, j in zip(train_df['num_options'], train_Y)]
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


	test_X = test_df.drop(columns=['ifp_id', 'date', 'answer', 'ordered', 'num_options']).drop(columns=drop_list).values
	test_Y = test_df['answer'].values
	test_Y_modulate = [i+j/10 for i, j in zip(test_df['num_options'], test_Y)]
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

	dtrain = xgb.DMatrix(train_X, train_Y_modulate, weight=train_W_modulate, feature_names=feature_names)
	dtest = xgb.DMatrix(test_X, test_Y_modulate, weight=test_W_modulate, feature_names=feature_names)

	param = {'eta': 0.1, 'gamma': 1, 'max_depth': 3, 'objective': 'multi:softprob', 'num_class': 5, 'verbosity': 3, 'lambda': 1, 'alpha': 1, 'disable_default_eval_metric': 1}
	watchlist = [(dtrain, 'train'), (dtest, 'eval')]

	bst = xgb.train(param, dtrain, num_boost_round=1000, evals=watchlist, obj=logregobj, feval=evalerror, early_stopping_rounds=50)
	bst.save_model('{}_{}.model'.format(task_name, counter))
	bst.dump_model('{}_{}.txt'.format(task_name, counter))
	print(bst.best_iteration, bst.best_ntree_limit)
	test_Y_prob = bst.predict(dtest, ntree_limit=bst.best_ntree_limit)

	test_Y_brier = []
	for i, v in enumerate(test_Y_prob):
		row = test_df.iloc[i]
		ifp_id = row['ifp_id']
		answer = row['answer']
		ordered = row['ordered']
		num_option = row['num_options']

		human_brier = brier_compute([row['h_1'], row['h_2'], row['h_3'], row['h_4'], row['h_5']][:num_option], answer, ordered)
		human1_brier = brier_compute([row['h1_1'], row['h1_2'], row['h1_3'], row['h1_4'], row['h1_5']][:num_option], answer, ordered)
		human2_brier = brier_compute([row['h2_1'], row['h2_2'], row['h2_3'], row['h2_4'], row['h2_5']][:num_option], answer, ordered)
		human3_brier = brier_compute([row['h3_1'], row['h3_2'], row['h3_3'], row['h3_4'], row['h3_5']][:num_option], answer, ordered)
		human4_brier = brier_compute([row['h4_1'], row['h4_2'], row['h4_3'], row['h4_4'], row['h4_5']][:num_option], answer, ordered)
		arima_brier = brier_compute([row['arima_1'], row['arima_2'], row['arima_3'], row['arima_4'], row['arima_5']][:num_option], answer, ordered)
		m4_brier = brier_compute([row['m4_1'], row['m4_2'], row['m4_3'], row['m4_4'], row['m4_5']][:num_option], answer, ordered)
		arw_brier = brier_compute([row['arw_1'], row['arw_2'], row['arw_3'], row['arw_4'], row['arw_5']][:num_option], answer, ordered)
		dsholt_brier = brier_compute([row['dsholt_1'], row['dsholt_2'], row['dsholt_3'], row['dsholt_4'], row['dsholt_5']][:num_option], answer, ordered)
		dsholtd_brier = brier_compute([row['dsholtd_1'], row['dsholtd_2'], row['dsholtd_3'], row['dsholtd_4'], row['dsholtd_5']][:num_option], answer, ordered)
		dsrw_brier = brier_compute([row['dsrw_1'], row['dsrw_2'], row['dsrw_3'], row['dsrw_4'], row['dsrw_5']][:num_option], answer, ordered)
		dsses_brier = brier_compute([row['dsses_1'], row['dsses_2'], row['dsses_3'], row['dsses_4'], row['dsses_5']][:num_option], answer, ordered)
		ets_brier = brier_compute([row['ets_1'], row['ets_2'], row['ets_3'], row['ets_4'], row['ets_5']][:num_option], answer, ordered)
		grw_brier = brier_compute([row['grw_1'], row['grw_2'], row['grw_3'], row['grw_4'], row['grw_5']][:num_option], answer, ordered)
		m4c_brier = brier_compute([row['m4c_1'], row['m4c_2'], row['m4c_3'], row['m4c_4'], row['m4c_5']][:num_option], answer, ordered)
		mean_brier = brier_compute([row['mean_1'], row['mean_2'], row['mean_3'], row['mean_4'], row['mean_5']][:num_option], answer, ordered)
		nnetar_brier = brier_compute([row['nnetar_1'], row['nnetar_2'], row['nnetar_3'], row['nnetar_4'], row['nnetar_5']][:num_option], answer, ordered)
		#rnn_brier = brier_compute([row['rnn_1'], row['rnn_2'], row['rnn_3'], row['rnn_4'], row['rnn_5']][:num_option], answer, ordered)
		rw_brier = brier_compute([row['rw_1'], row['rw_2'], row['rw_3'], row['rw_4'], row['rw_5']][:num_option], answer, ordered)
		rwd_brier = brier_compute([row['rwd_1'], row['rwd_2'], row['rwd_3'], row['rwd_4'], row['rwd_5']][:num_option], answer, ordered)
		rws_brier = brier_compute([row['rws_1'], row['rws_2'], row['rws_3'], row['rws_4'], row['rws_5']][:num_option], answer, ordered)
		#sim_brier = brier_compute([row['sim_1'], row['sim_2'], row['sim_3'], row['sim_4'], row['sim_5']], answer, ordered)
		sltmar_brier = brier_compute([row['sltmar_1'], row['sltmar_2'], row['sltmar_3'], row['sltmar_4'], row['sltmar_5']][:num_option], answer, ordered)
		tbats_brier = brier_compute([row['tbats_1'], row['tbats_2'], row['tbats_3'], row['tbats_4'], row['tbats_5']][:num_option], answer, ordered)
		theta_brier = brier_compute([row['theta_1'], row['theta_2'], row['theta_3'], row['theta_4'], row['theta_5']][:num_option], answer, ordered)

		prob = v[:num_option]
		prob /= sum(prob)

		predict_brier = brier_compute(prob, answer, ordered)
		test_Y_brier.append([human_brier, human1_brier, human2_brier, human3_brier, human4_brier, arima_brier, m4_brier, arw_brier, dsholt_brier, dsholtd_brier, dsrw_brier, dsses_brier, ets_brier, grw_brier, m4c_brier, mean_brier, nnetar_brier, rw_brier, rwd_brier, rws_brier, sltmar_brier, tbats_brier, theta_brier, predict_brier])
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
	splits.append([briers_mean[0],  briers_mean[5], briers_mean[-3], briers_mean[-1]])
	counter += 1

	'''
	explainer = shap.TreeExplainer(bst)
	shap_values = np.asarray(explainer.shap_values(dtrain))

	shap.force_plot(explainer.expected_value[0], shap_values[0], matplotlib=True)
	pdb.set_trace()
	'''




splits = np.asarray(splits)
print('human', 'arima', 'two_ideal', 'predict')
print(np.mean(splits, axis=0))
print(splits)
pdb.set_trace()
print('OK')
