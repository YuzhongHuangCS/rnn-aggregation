import tensorflow as tf
import pickle
import pandas as pd
import numpy as np
import pdb
from briercompute import brier as brier_compute
import os

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
train_Y = np.zeros((len(train_X), 5))
for i, v in enumerate(train_df['answer'].values):
	train_Y[i, v] = 1
train_Z = train_df['ordered'].values.astype(int)

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

test_X = test_df.drop(columns=['ifp_id', 'date', 'answer', 'ordered']).drop(columns=drop_list).values
test_Y = np.zeros((len(test_X), 5))
for i, v in enumerate(test_df['answer'].values):
	test_Y[i, v] = 1
test_Z = test_df['ordered'].values

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

n_features = train_X.shape[1]
n_hidden1 = 64
print('n_features', n_features)

X_placeholder = tf.placeholder(tf.float32, (None, n_features), name='X')
Y_placeholder = tf.placeholder(tf.float32, (None, 5), name='Y')
Z_placeholder = tf.placeholder(tf.bool, (None, ), name='Z')
W_placeholder = tf.placeholder(tf.float32, (None, ), name='W')

W1 = tf.get_variable('W1', shape=(n_features, n_hidden1), initializer=tf.glorot_uniform_initializer())
b1 = tf.get_variable('b1', shape=(1, n_hidden1), initializer=tf.zeros_initializer())
W2 = tf.get_variable('W2', shape=(n_hidden1, 5), initializer=tf.glorot_uniform_initializer())
b2 = tf.get_variable('b2', shape=(1, 5), initializer=tf.zeros_initializer())

l_output = ttf.matmul(tf.nn.tanh(tf.matmul(X_placeholder, W1) + b1), W2) + b2
output = tf.nn.softmax(l_output)

lside_1 = tf.reduce_sum(tf.gather(output, [0], axis=1), axis=1)
rside_1 = tf.reduce_sum(tf.gather(output, [1, 2, 3, 4], axis=1), axis=1)
output_1 = tf.stack([lside_1, rside_1], axis=1)
lside_true_1 = tf.reduce_sum(tf.gather(Y_placeholder, [0], axis=1), axis=1)
rside_true_1 = tf.reduce_sum(tf.gather(Y_placeholder, [1, 2, 3, 4], axis=1), axis=1)
true_1 = tf.stack([lside_true_1, rside_true_1], axis=1)
loss_1 = tf.math.reduce_sum(tf.math.squared_difference(output_1, true_1), axis=1)

lside_2 = tf.reduce_sum(tf.gather(output, [0, 1], axis=1), axis=1)
rside_2 = tf.reduce_sum(tf.gather(output, [2, 3, 4], axis=1), axis=1)
output_2 = tf.stack([lside_2, rside_2], axis=1)
lside_true_2 = tf.reduce_sum(tf.gather(Y_placeholder, [0, 1], axis=1), axis=1)
rside_true_2 = tf.reduce_sum(tf.gather(Y_placeholder, [2, 3, 4], axis=1), axis=1)
true_2 = tf.stack([lside_true_2, rside_true_2], axis=1)
loss_2 = tf.math.reduce_sum(tf.math.squared_difference(output_2, true_2), axis=1)

lside_3 = tf.reduce_sum(tf.gather(output, [0, 1, 2], axis=1), axis=1)
rside_3 = tf.reduce_sum(tf.gather(output, [3, 4], axis=1), axis=1)
output_3 = tf.stack([lside_3, rside_3], axis=1)
lside_true_3 = tf.reduce_sum(tf.gather(Y_placeholder, [0, 1, 2], axis=1), axis=1)
rside_true_3 = tf.reduce_sum(tf.gather(Y_placeholder, [3, 4], axis=1), axis=1)
true_3 = tf.stack([lside_true_3, rside_true_3], axis=1)
loss_3 = tf.math.reduce_sum(tf.math.squared_difference(output_3, true_3), axis=1)

lside_4 = tf.reduce_sum(tf.gather(output, [0, 1, 2, 3], axis=1), axis=1)
rside_4 = tf.reduce_sum(tf.gather(output, [4], axis=1), axis=1)
output_4 = tf.stack([lside_4, rside_4], axis=1)

lside_true_4 = tf.reduce_sum(tf.gather(Y_placeholder, [0, 1, 2, 3], axis=1), axis=1)
rside_true_4 = tf.reduce_sum(tf.gather(Y_placeholder, [4], axis=1), axis=1)
true_4 = tf.stack([lside_true_4, rside_true_4], axis=1)
loss_4 = tf.math.reduce_sum(tf.math.squared_difference(output_4, true_4), axis=1)

loss_brier = tf.math.reduce_mean(tf.stack([loss_1, loss_2, loss_3, loss_4], axis=1), axis=1)
#loss_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y_placeholder, logits=l_output)
loss_mse = tf.math.reduce_sum(tf.math.squared_difference(Y_placeholder, output), axis=1)
#loss_mse = tf.losses.mean_squared_error(Y_placeholder, output)
#pdb.set_trace()
loss_combined = tf.where(Z_placeholder, loss_brier, loss_mse)
#pdb.set_trace()
loss_combined_reduce = tf.losses.compute_weighted_loss(loss_combined, W_placeholder)
#loss_combined_reduce = tf.math.reduce_mean(loss_combined)

loss_combined_reduce_reg = loss_combined_reduce
variables = [v for v in tf.trainable_variables() if 'W' in v.name]


for v in variables:
	loss_combined_reduce_reg += 0.0002 * tf.nn.l2_loss(v) + 0.002 * tf.losses.absolute_difference(v, tf.zeros(tf.shape(v)))
	#loss_combined_reduce_reg += 0.05 * tf.nn.l2_loss(v)

train_op = tf.train.AdamOptimizer(0.005).minimize(loss=loss_combined_reduce_reg)
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	loss_list = []
	for epoch in range(50):
		train_step, loss_combined_reduce_train = sess.run(
		[train_op, loss_combined_reduce],
		feed_dict={
			X_placeholder: train_X,
			Y_placeholder: train_Y,
			Z_placeholder: train_Z,
			W_placeholder: train_W
		})

		#pdb.set_trace()
		[test_Y_prob, loss_combined_reduce_test] = sess.run(
		[output, loss_combined_reduce],
		feed_dict={
			X_placeholder: test_X,
			Y_placeholder: test_Y,
			Z_placeholder: test_Z,
			W_placeholder: test_W
		})

		print(epoch, loss_combined_reduce_train, loss_combined_reduce_test)

		if epoch == 0:
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
				#pdb.set_trace()
				two_ideal = np.mean(np.min([test_Y_brier[index][:, 0], test_Y_brier[index][:, 5]], axis=0))
				ideal = np.mean(np.min(test_Y_brier[index][:, :-1], axis=1))
				scores.insert(-1, two_ideal)
				scores.insert(-1, ideal)
				db_brier[ifp] = np.asarray(scores)

			briers = np.asarray(list(db_brier.values()))
			#pdb.set_trace()
			briers_mean = np.mean(briers, axis=0)
			print('human', 'arima', 'two_ideal', 'ideal', 'predict', 'loss')
			print(briers_mean[0], briers_mean[5], briers_mean[-3], briers_mean[-2], briers_mean[-1], loss_combined_reduce_test)
		loss_list.append(loss_combined_reduce_test)

	print('min loss', np.min(loss_list))

	'''
	with open('briers.csv', 'w') as fout:
		fout.write('ifp, human, arima, m4, arw, dsholt, dsholtd, dsrw, dsses, ets, grw, m4c, mean, nnetar, rnn, rw, rwd, rws, sim, sltmar, tbats, theta, two_ideal, ideal, predict\n')
		for ifp, row in db_brier.items():
			fout.write(str(ifp) + ',' + ','.join([str(x) for x in row]) + '\n')
	'''
