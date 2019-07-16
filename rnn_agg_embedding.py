import os

# uncomment to force CPU training
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# optimize CPU performance
os.environ['KMP_BLOCKTIME'] = '0'
os.environ['KMP_AFFINITY'] = 'granularity=fine,verbose,compact,1,0'

import pandas as pd
import pdb
import dateutil.parser
import numpy as np
from collections import OrderedDict, Counter
import pickle
import sklearn.model_selection
import sklearn.decomposition
import sklearn.cluster
import scipy.stats
import tensorflow as tf
import random
import copy
from briercompute import brier
from datetime import datetime, timedelta
import math

def is_ordered(opt):
	keywords = ['Less', 'Between', 'More', 'inclusive','less', 'between', 'more']
	if len(opt) <= 2: #binary
		return False
	for o in opt:
		if any(x in o for x in keywords):
			return True

	return False

db_answer = {}
db_dates = {}
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

					start_date = dateutil.parser.parse(row['start_date']).replace(hour=0, minute=0, second=0, microsecond=0)
					end_date = dateutil.parser.parse(row['end_date']).replace(hour=0, minute=0, second=0, microsecond=0)

					forecast_dates = []
					forecast_date = start_date
					while forecast_date <= end_date:
						forecast_dates.append(forecast_date)
						forecast_date += timedelta(days=1)
					db_dates[ifp_id] = forecast_dates
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
all_ifp = np.asarray(list(db.keys()))

kf = sklearn.model_selection.KFold(shuffle=True, n_splits=5, random_state=2019)
folds = [[all_ifp[f[0]], all_ifp[f[1]]] for f in kf.split(all_ifp)]
fold_index = 0

ifp_train = folds[fold_index][0]
ifp_test = folds[fold_index][1]

n_train = len(ifp_train)
n_test = len(ifp_test)

N_RNN_DIM = 32
#N_PROJ_DIM = 16
N_EMB_DIM = 8

special_symbol = {
	'padding': 0,
	'unknown': 1
}

id_counter = Counter()
id_counter.update(df[df['ifp_id'].isin(ifp_train)]['user_id'])
id2index = {}
index2id = {}
for index, value in enumerate(id_counter.most_common()):
	id2index[value[0]] = index + len(special_symbol)
	index2id[index + len(special_symbol)] = value[0]

### TRAIN data
n_forecast_train = sum([len(v) for k, v in db_dates.items() if k in ifp_train])

input_train = np.zeros((n_train, max_steps, 5))
id_train = np.zeros((n_train, max_steps, 1), dtype=int)
target_train = np.zeros((n_forecast_train, 5))
answer_train = np.zeros(n_forecast_train, dtype=int)
is_ordered_train = np.zeros(n_forecast_train, dtype=bool)
is_4_train = np.zeros(n_forecast_train, dtype=bool)
is_3_train = np.zeros(n_forecast_train, dtype=bool)
weight_train = np.zeros(n_forecast_train)
seq_length_train = np.zeros(n_train, dtype=int)
gather_index_train = np.zeros((n_forecast_train, 2), dtype=int)
num_option_ary_train = np.zeros(n_forecast_train, dtype=int)
num_option_mask_train = np.zeros((n_forecast_train, 5))
index_map_train = {}

forecast_index = 0
for index, ifp in enumerate(ifp_train):
	forecasts = db[ifp]

	for i, forecast in enumerate(forecasts):
		input_train[index, i] = forecast[-5:]

		forecaster_id = id2index.get(forecast[1], 1)
		id_train[index, i] = forecaster_id

	forecast_dates = db_dates[ifp]
	n_forecasts = len(forecast_dates)
	activity_dates = [dateutil.parser.parse(x[0]) for x in forecasts]

	answer, is_ordered = db_answer[ifp]
	target_train[forecast_index:forecast_index+n_forecasts, answer] = 1
	answer_train[forecast_index:forecast_index+n_forecasts] = answer
	is_ordered_train[forecast_index:forecast_index+n_forecasts] = is_ordered
	weight_train[forecast_index:forecast_index+n_forecasts] = ((1.0 / n_forecasts) / n_train) * n_forecast_train
	seq_length_train[index] = len(forecasts)

	for i, forecast_date in enumerate(forecast_dates):
		this_index = np.searchsorted(activity_dates, forecast_date)
		# hack here! If no forecast on the first day, use the first subsequent
		i2 = i
		while this_index == 0:
			this_index = np.searchsorted(activity_dates, forecast_dates[i2+1])
			i2 += 1
		gather_index_train[forecast_index+i, :] = [index, this_index-1]

	num_options = forecasts[0][3]
	if num_options == 4:
		is_4_train[forecast_index:forecast_index+n_forecasts] = True
	else:
		if num_options == 3:
			is_3_train[forecast_index:forecast_index+n_forecasts] = True

	num_option_ary_train[forecast_index:forecast_index+n_forecasts] = num_options
	num_option_mask_train[forecast_index:forecast_index+n_forecasts, :num_options] = 1

	index_map_train[ifp] = list(range(forecast_index, forecast_index+n_forecasts))
	forecast_index += n_forecasts

input_train[np.isnan(input_train)] = 0

### TEST data
n_forecast_test = sum([len(v) for k, v in db_dates.items() if k in ifp_test])

input_test = np.zeros((n_test, max_steps, 5))
id_test = np.zeros((n_test, max_steps, 1), dtype=int)
target_test = np.zeros((n_forecast_test, 5))
answer_test = np.zeros(n_forecast_test, dtype=int)
is_ordered_test = np.zeros(n_forecast_test, dtype=bool)
is_4_test = np.zeros(n_forecast_test, dtype=bool)
is_3_test = np.zeros(n_forecast_test, dtype=bool)
weight_test = np.zeros(n_forecast_test)
seq_length_test = np.zeros(n_test, dtype=int)
gather_index_test = np.zeros((n_forecast_test, 2), dtype=int)
num_option_ary_test = np.zeros(n_forecast_test, dtype=int)
num_option_mask_test = np.zeros((n_forecast_test, 5))
index_map_test = {}

forecast_index = 0
for index, ifp in enumerate(ifp_test):
	forecasts = db[ifp]

	for i, forecast in enumerate(forecasts):
		input_test[index, i] = forecast[-5:]

		forecaster_id = id2index.get(forecast[1], 1)
		id_test[index, i] = forecaster_id

	forecast_dates = db_dates[ifp]
	n_forecasts = len(forecast_dates)
	activity_dates = [dateutil.parser.parse(x[0]) for x in forecasts]

	answer, is_ordered = db_answer[ifp]
	target_test[forecast_index:forecast_index+n_forecasts, answer] = 1
	answer_test[forecast_index:forecast_index+n_forecasts] = answer
	is_ordered_test[forecast_index:forecast_index+n_forecasts] = is_ordered
	weight_test[forecast_index:forecast_index+n_forecasts] = ((1.0 / n_forecasts) / n_test) * n_forecast_test
	seq_length_test[index] = len(forecasts)

	for i, forecast_date in enumerate(forecast_dates):
		this_index = np.searchsorted(activity_dates, forecast_date)
		# hack here! If no forecast on the first day, use the first subsequent
		i2 = i
		while this_index == 0:
			this_index = np.searchsorted(activity_dates, forecast_dates[i2+1])
			i2 += 1
		gather_index_test[forecast_index+i, :] = [index, this_index-1]

	num_options = forecasts[0][3]
	if num_options == 4:
		is_4_test[forecast_index:forecast_index+n_forecasts] = True
	else:
		if num_options == 3:
			is_3_test[forecast_index:forecast_index+n_forecasts] = True

	num_option_ary_test[forecast_index:forecast_index+n_forecasts] = num_options
	num_option_mask_test[forecast_index:forecast_index+n_forecasts, :num_options] = 1

	index_map_test[ifp] = list(range(forecast_index, forecast_index+n_forecasts))
	forecast_index += n_forecasts

input_test[np.isnan(input_test)] = 0


# Network placeholder
input_placeholder = tf.placeholder(tf.float32, [None, max_steps, 5])
id_placeholder = tf.placeholder(tf.int32, [None, max_steps, 1])
target_placeholder = tf.placeholder(tf.float32, [None, 5])
is_ordered_placeholder = tf.placeholder(tf.bool, [None])
is_4_placeholder = tf.placeholder(tf.bool, [None])
is_3_placeholder = tf.placeholder(tf.bool, [None])
weight_placeholder = tf.placeholder(tf.float32, [None])
seq_length_placeholder = tf.placeholder(tf.int32, [None])
gather_index_placeholder = tf.placeholder(tf.int32, [None, 2])
num_option_mask_placeholder = tf.placeholder(tf.float32, [None, 5])

id_size = len(id2index) + len(special_symbol)
embedding = tf.get_variable('embedding', shape=(id_size, N_EMB_DIM), initializer=tf.random_uniform_initializer(-math.sqrt(3), math.sqrt(3)))
embedded_features = tf.nn.embedding_lookup(embedding, id_placeholder)

combined_input = tf.concat([input_placeholder, embedded_features[:, :, 0, :]], 2)
#cell = tf.nn.rnn_cell.LSTMCell(N_RNN_DIM, use_peepholes=True, num_proj=N_PROJ_DIM)
cell = tf.nn.rnn_cell.LSTMCell(N_RNN_DIM, initializer=tf.orthogonal_initializer())
#cell = tf.nn.rnn_cell.GRUCell(N_RNN_DIM, kernel_initializer=tf.orthogonal_initializer(), bias_initializer=tf.zeros_initializer())
state_series, _ = tf.nn.dynamic_rnn(cell, combined_input, sequence_length=seq_length_placeholder, dtype=tf.float32)
W1 = tf.get_variable('weight1', shape=(N_RNN_DIM, 5), initializer=tf.glorot_uniform_initializer())
b1 = tf.get_variable('bias1', shape=(1, 5), initializer=tf.zeros_initializer())
needed_state = tf.gather_nd(state_series, gather_index_placeholder)
prediction = tf.matmul(tf.tanh(needed_state), W1) + b1
prediction_softmax = tf.nn.softmax(prediction)
raw_prob = tf.math.multiply(prediction_softmax, num_option_mask_placeholder)
prob_row_sum = tf.reduce_sum(raw_prob, axis=1)
prob = tf.div(raw_prob, tf.reshape(prob_row_sum, (-1, 1)))
loss_mse = tf.math.reduce_sum(tf.math.squared_difference(target_placeholder, prob), axis=1)

prob_1 = tf.stack([tf.reduce_sum(tf.gather(prob, [0], axis=1), axis=1), tf.reduce_sum(tf.gather(prob, [1, 2, 3, 4], axis=1), axis=1)], axis=1)
true_1 = tf.stack([tf.reduce_sum(tf.gather(target_placeholder, [0], axis=1), axis=1), tf.reduce_sum(tf.gather(target_placeholder, [1, 2, 3, 4], axis=1), axis=1)], axis=1)
loss_1 = tf.math.reduce_sum(tf.math.squared_difference(prob_1, true_1), axis=1)

prob_2 = tf.stack([tf.reduce_sum(tf.gather(prob, [0, 1], axis=1), axis=1), tf.reduce_sum(tf.gather(prob, [2, 3, 4], axis=1), axis=1)], axis=1)
true_2 = tf.stack([tf.reduce_sum(tf.gather(target_placeholder, [0, 1], axis=1), axis=1), tf.reduce_sum(tf.gather(target_placeholder, [2, 3, 4], axis=1), axis=1)], axis=1)
loss_2 = tf.math.reduce_sum(tf.math.squared_difference(prob_2, true_2), axis=1)

prob_3 = tf.stack([tf.reduce_sum(tf.gather(prob, [0, 1, 2], axis=1), axis=1), tf.reduce_sum(tf.gather(prob, [3, 4], axis=1), axis=1)], axis=1)
true_3 = tf.stack([tf.reduce_sum(tf.gather(target_placeholder, [0, 1, 2], axis=1), axis=1), tf.reduce_sum(tf.gather(target_placeholder, [3, 4], axis=1), axis=1)], axis=1)
loss_3 = tf.math.reduce_sum(tf.math.squared_difference(prob_3, true_3), axis=1)

prob_4 = tf.stack([tf.reduce_sum(tf.gather(prob, [0, 1, 2, 3], axis=1), axis=1), tf.reduce_sum(tf.gather(prob, [4], axis=1), axis=1)], axis=1)
true_4 = tf.stack([tf.reduce_sum(tf.gather(target_placeholder, [0, 1, 2, 3], axis=1), axis=1), tf.reduce_sum(tf.gather(target_placeholder, [4], axis=1), axis=1)], axis=1)
loss_4 = tf.math.reduce_sum(tf.math.squared_difference(prob_4, true_4), axis=1)

loss_brier = tf.math.reduce_mean(tf.stack([loss_1, loss_2, loss_3, loss_4], axis=1), axis=1)


prob_4_1 = tf.stack([tf.reduce_sum(tf.gather(prob, [0], axis=1), axis=1), tf.reduce_sum(tf.gather(prob, [1, 2, 3], axis=1), axis=1)], axis=1)
true_4_1 = tf.stack([tf.reduce_sum(tf.gather(target_placeholder, [0], axis=1), axis=1), tf.reduce_sum(tf.gather(target_placeholder, [1, 2, 3], axis=1), axis=1)], axis=1)
loss_4_1 = tf.math.reduce_sum(tf.math.squared_difference(prob_4_1, true_4_1), axis=1)

prob_4_2 = tf.stack([tf.reduce_sum(tf.gather(prob, [0, 1], axis=1), axis=1), tf.reduce_sum(tf.gather(prob, [2, 3], axis=1), axis=1)], axis=1)
true_4_2 = tf.stack([tf.reduce_sum(tf.gather(target_placeholder, [0, 1], axis=1), axis=1), tf.reduce_sum(tf.gather(target_placeholder, [2, 3], axis=1), axis=1)], axis=1)
loss_4_2 = tf.math.reduce_sum(tf.math.squared_difference(prob_4_2, true_4_2), axis=1)

prob_4_3 = tf.stack([tf.reduce_sum(tf.gather(prob, [0, 1, 2], axis=1), axis=1), tf.reduce_sum(tf.gather(prob, [3], axis=1), axis=1)], axis=1)
true_4_3 = tf.stack([tf.reduce_sum(tf.gather(target_placeholder, [0, 1, 2], axis=1), axis=1), tf.reduce_sum(tf.gather(target_placeholder, [3], axis=1), axis=1)], axis=1)
loss_4_3 = tf.math.reduce_sum(tf.math.squared_difference(prob_4_3, true_4_3), axis=1)

loss_brier_4 = tf.math.reduce_mean(tf.stack([loss_4_1, loss_4_2, loss_4_3], axis=1), axis=1)

prob_3_1 = tf.stack([tf.reduce_sum(tf.gather(prob, [0], axis=1), axis=1), tf.reduce_sum(tf.gather(prob, [1, 2], axis=1), axis=1)], axis=1)
true_3_1 = tf.stack([tf.reduce_sum(tf.gather(target_placeholder, [0], axis=1), axis=1), tf.reduce_sum(tf.gather(target_placeholder, [1, 2], axis=1), axis=1)], axis=1)
loss_3_1 = tf.math.reduce_sum(tf.math.squared_difference(prob_3_1, true_3_1), axis=1)

prob_3_2 = tf.stack([tf.reduce_sum(tf.gather(prob, [0, 1], axis=1), axis=1), tf.reduce_sum(tf.gather(prob, [2], axis=1), axis=1)], axis=1)
true_3_2 = tf.stack([tf.reduce_sum(tf.gather(target_placeholder, [0, 1], axis=1), axis=1), tf.reduce_sum(tf.gather(target_placeholder, [2], axis=1), axis=1)], axis=1)
loss_3_2 = tf.math.reduce_sum(tf.math.squared_difference(prob_3_2, true_3_2), axis=1)

loss_brier_3 = tf.math.reduce_mean(tf.stack([loss_3_1, loss_3_2], axis=1), axis=1)

#loss_combined = tf.where(is_ordered_placeholder, loss_brier, loss_mse)
loss_combined = tf.where(is_ordered_placeholder, tf.where(is_4_placeholder, loss_brier_4, tf.where(is_3_placeholder, loss_brier_3, loss_brier)), loss_mse)

loss_weighted = tf.losses.compute_weighted_loss(loss_combined, weight_placeholder)


loss_weighted_reg = loss_weighted
variables = [v for v in tf.trainable_variables() if 'bias' not in v.name]

for v in variables:
	loss_weighted_reg += 0.01 * tf.nn.l2_loss(v) + 0.01 * tf.losses.absolute_difference(v, tf.zeros(tf.shape(v)))

train_op = tf.train.AdamOptimizer(0.01).minimize(loss=loss_weighted_reg)
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	test_scores = []
	for i in range(100):
		train_loss, train_pred, _train_step = sess.run(
			[loss_weighted, prob, train_op],
				feed_dict={
					input_placeholder: input_train,
					id_placeholder: id_train,
					target_placeholder: target_train,
					is_ordered_placeholder: is_ordered_train,
					is_4_placeholder: is_4_train,
					is_3_placeholder: is_3_train,
					weight_placeholder: weight_train,
					seq_length_placeholder: seq_length_train,
					gather_index_placeholder: gather_index_train,
					num_option_mask_placeholder: num_option_mask_train
				}
		)

		test_loss, test_pred = sess.run(
			[loss_weighted, prob],
				feed_dict={
					input_placeholder: input_test,
					id_placeholder: id_test,
					target_placeholder: target_test,
					is_ordered_placeholder: is_ordered_test,
					is_4_placeholder: is_4_test,
					is_3_placeholder: is_3_test,
					weight_placeholder: weight_test,
					seq_length_placeholder: seq_length_test,
					gather_index_placeholder: gather_index_test,
					num_option_mask_placeholder: num_option_mask_test
				}
		)

		print(i, train_loss, test_loss)
		test_scores.append(test_loss)
		if i == 0:
			train_briers = np.asarray([brier(p[:num_option_ary_train[i]], answer_train[i], is_ordered_train[i]) for i, p in enumerate(train_pred)])
			test_briers = np.asarray([brier(p[:num_option_ary_test[i]], answer_test[i], is_ordered_test[i]) for i, p in enumerate(test_pred)])

			db_brier_train = {}
			for ifp in ifp_train:
				index = index_map_train[ifp]
				scores = train_briers[index]
				db_brier_train[ifp] = np.mean(scores)

			db_brier_test = {}
			for ifp in ifp_test:
				index = index_map_test[ifp]
				scores = test_briers[index]
				db_brier_test[ifp] = np.mean(scores)

			print(i, train_loss, test_loss, np.mean(list(db_brier_train.values())), np.mean(list(db_brier_test.values())))

	print('min test loss', np.min(test_scores))
print('OK')