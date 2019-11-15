import os

# uncomment to force CPU training
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# optimize CPU performance
#os.environ['KMP_BLOCKTIME'] = '0'
#os.environ['KMP_AFFINITY'] = 'granularity=fine,verbose,compact,1,0'

import pandas as pd
import pdb
import sys
import dateutil.parser
import numpy as np
from collections import OrderedDict, Counter, defaultdict
import pickle
import sklearn.model_selection
import tensorflow as tf
import random
import copy
from briercompute import brier
from datetime import datetime, timedelta
import math
from nltk.tokenize import word_tokenize
from utils import is_ordered, initialize_embedding, embedding_lookup

if len(sys.argv) >= 3:
	model_name = sys.argv[1]
	fold_index = int(sys.argv[2])
	feature_used = sys.argv[3:]
else:
	model_name = 'att_model'
	fold_index = 0
	feature_used = ['None', ]

print('fold_index: {}'.format(fold_index))
print('feature_used: {}'.format(feature_used))

word_embedding = initialize_embedding()
db_answer = {}
db_boundary = {}
db_emb = {}

# 1. db_date is first computed by question dump
# 2. db_date is then adjusted by actual forecasts
discover2ifp = {}
for filename in ('data/dump_questions_rcta.csv', 'data/dump_questions_rctb.csv', 'data/dump_questions_rctc.csv'):
	df_question = pd.read_csv(filename)
	for index, row in df_question.iterrows():
		if filename == 'data/dump_questions_rctc.csv':
			ifp_id = row['hfc_id']
			discover_id = row['discover_id']
			discover2ifp[discover_id] = ifp_id
		else:
			ifp_id = row['ifp_id']

		if row['is_resolved'] and (not row['is_voided']):
			resolution = row['resolution']
			options = row.tolist()[-5:]

			clean_options = [x for x in options if type(x) == str]
			try:
				answer = options.index(resolution)
				if ifp_id in db_answer:
					pdb.set_trace()
					print('Duplicated ifp')
				else:
					db_answer[ifp_id] = [answer, is_ordered(clean_options)]

					if filename == 'data/dump_questions_rcta.csv':
						start_date = dateutil.parser.parse(row['start_date']).replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)
						end_date = dateutil.parser.parse(row['end_date']).replace(hour=23, minute=59, second=59, microsecond=999, tzinfo=None)
					else:
						start_date = dateutil.parser.parse(row['scoring_start_time']).replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)
						end_date = dateutil.parser.parse(row['scoring_end_time']).replace(hour=23, minute=59, second=59, microsecond=999, tzinfo=None)

					resolved_date = dateutil.parser.parse(row['resolved_date']).replace(hour=23, minute=59, second=59, microsecond=999, tzinfo=None)
					end_date = min(end_date, resolved_date)
					db_boundary[ifp_id] = [start_date, end_date]

					features = []
					for w in word_tokenize(row['title']):
						feature = embedding_lookup(w, word_embedding)
						if feature is not None:
							features.append(feature)

					feature_mean = np.mean(np.asarray(features), axis=0)
					db_emb[ifp_id] = feature_mean

			except ValueError as e:
				pdb.set_trace()
				print(e)

human_feature_list = ['n_forecasts_te', 'variance_sage', 'n_forecasts_d', 'n_forecasts_sage', 'entropy_b', 'entropy_d', 'entropy_te', 'n_forecasts_b', 'entropy_c', 'Technology', 'variance_b', 'variance_d', 'Other', 'n_forecasts_c', 'stage', 'entropy_sage', 'n_forecasts', 'Politics/Intl Relations', 'Macroeconomics/Finance', 'Health/Disease', 'variance_te', 'variance_c', 'variance_human', 'entropy_human', 'ordinal', 'Natural Sciences/Climate', 'p_updates']
human_feature = pd.read_csv('data/human_features.csv').drop_duplicates(subset=['date', 'ifp_id'], keep='last')
hf_cols = human_feature.columns.tolist()
hf_cols.remove('ifp_id')
hf_cols.remove('date')
hf_cols.remove('p_updates')
hf_cols.remove('Health/Disease')
if feature_used is not None:
	for fu in feature_used:
		if fu in human_feature_list and fu in hf_cols:
			hf_cols.remove(fu)

human_feature = human_feature.drop(columns=hf_cols)
human_feature_rctc = pd.read_csv('data/human_features_rctc.csv').drop_duplicates(subset=['date', 'ifp_id'], keep='last')[human_feature.columns]

ts_feature_list = ['diff2_acf10', 'entropy', 'diff1_acf10', 'seas_pacf', 'linearity', 'spike', 'nonlinearity', 'diff1x_pacf5', 'e_acf10', 'series_length', 'hurst', 'ARCH.LM', 'ratio', 'seas_acf1', 'x_acf1', 'crossing_points', 'x_pacf5', 'diff1_acf1', 'trend', 'trough', 'unitroot_pp', 'diff2x_pacf5', 'x_acf10', 'nperiods', 'flat_spots', 'seasonal_period', 'peak', 'beta', 'diff2_acf1', 'lumpiness', 'e_acf1', 'skew', 'curvature', 'alpha', 'unitroot_kpss', 'seasonal_strength', 'stability']
ts_feature_rctc = pd.read_csv('data/ts_features_rctc.csv')
tf_cols = ts_feature_rctc.columns.tolist()
tf_cols.remove('ifp_id')
tf_cols.remove('date')
tf_cols.remove('ratio')
if feature_used is not None:
	for fu in feature_used:
		if fu in ts_feature_list and fu in tf_cols:
			tf_cols.remove(fu)
ts_feature_rctc = ts_feature_rctc.drop(columns=tf_cols)
ts_feature = pd.read_csv('data/ts_features.csv')[ts_feature_rctc.columns]

n_feature = human_feature.shape[1] + ts_feature.shape[1] - 4
print(human_feature.columns)
print(ts_feature.columns)
print('n_feature', n_feature)

human_dict = defaultdict(dict)
for h_f in (human_feature, human_feature_rctc):
	for index, row in h_f.iterrows():
		ifp_id = row['ifp_id']
		date = row['date']

		if date in human_dict[ifp_id]:
			pdb.set_trace()
			print('Duplicate feature')
		else:
			human_dict[ifp_id][date] = row.drop(labels=['ifp_id', 'date']).values

ts_dict = defaultdict(dict)
for index, row in ts_feature.iterrows():
	ifp_id = row['ifp_id']
	date = row['date']

	if date in ts_dict[ifp_id]:
		pdb.set_trace()
		print('Duplicate feature')
	else:
		ts_dict[ifp_id][date] = row.drop(labels=['ifp_id', 'date']).values


for index, row in ts_feature_rctc.iterrows():
	discover_id = row['ifp_id']
	try:
		ifp_id = discover2ifp[discover_id]
	except KeyError as e:
		pdb.set_trace()
	date = row['date']

	if date in ts_dict[ifp_id]:
		pdb.set_trace()
		print('Duplicate feature')
	else:
		ts_dict[ifp_id][date] = row.drop(labels=['ifp_id', 'date']).values

def get_feature(ifp_id, date):
	if ifp_id in human_dict and date in human_dict[ifp_id]:
		hf = human_dict[ifp_id][date]
	else:
		hf = np.zeros(human_feature.shape[1]-2)

	if ifp_id in ts_dict and date in ts_dict[ifp_id]:
		mf = ts_dict[ifp_id][date]
	else:
		mf = np.zeros(ts_feature.shape[1]-2)

	try:
		cf = np.concatenate([hf, mf])
	except ValueError as e:
		pdb.set_trace()
		print('OK')

	return cf

db = defaultdict(list)
for filename in ('data/dump_user_forecasts_rcta.csv', 'data/dump_user_forecasts_rctb.csv', 'data/dump_user_forecasts_rctc.csv'):
	df = pd.read_csv(filename)
	for index, row in df.iterrows():
		date = dateutil.parser.parse(row['date']).replace(tzinfo=None)

		if filename == 'data/dump_user_forecasts_rctc.csv':
			ifp_id = row['hfc_id']
		else:
			ifp_id = row['ifp_id']

		if ifp_id not in db_answer:
			continue

		user_id = row['user_id']
		if filename == 'data/dump_user_forecasts_rctb.csv':
			user_id += 100000
		elif filename == 'data/dump_user_forecasts_rctc.csv':
			user_id += 200000

		num_options = row['num_options']
		option_1 = row['option_1'] / 100.0
		option_2 = row['option_2'] / 100.0
		option_3 = row['option_3'] / 100.0
		option_4 = row['option_4'] / 100.0
		option_5 = row['option_5'] / 100.0

		if num_options == 1:
			num_options = 2

		cf = get_feature(ifp_id, datetime.strftime(date, "%Y-%m-%d"))
		db[ifp_id].append([date,user_id,ifp_id,num_options,option_1,option_2,option_3,option_4,option_5] + cf.tolist())

machine_df = pd.read_csv('data/machine_all.csv').drop_duplicates(subset=['date', 'machine_model', 'ifp_id'], keep='last')
for index, row in machine_df.iterrows():
	date = dateutil.parser.parse(row['date']).replace(tzinfo=None)
	machine_model = row['machine_model']
	ifp_id = row['ifp_id']

	if ifp_id not in db_answer:
		continue

	if machine_model not in ('Auto ARIMA', 'M4-Meta', 'Arithmetic RW', ):
		continue

	if machine_model == 'M4-Meta':
		date = date.replace(microsecond=1)

	if machine_model == 'Arithmetic RW':
		date = date.replace(microsecond=2)

	num_options = row['num_options']
	option_1 = row['option_1']
	option_2 = row['option_2']
	option_3 = row['option_3']
	option_4 = row['option_4']
	option_5 = row['option_5']

	if num_options == 0:
		continue

	if num_options == 1:
		num_options = 2

	if ifp_id not in db:
		pdb.set_trace()
		print("Didn't expect any ifp have human forecast but don't have machine forecast")

	cf = get_feature(ifp_id, datetime.strftime(date, "%Y-%m-%d"))
	db[ifp_id].append([date,machine_model,ifp_id,num_options,option_1,option_2,option_3,option_4,option_5] + cf.tolist())

db_dates = {}
deleted_ifp = []
for ifp_id in db:
	db[ifp_id].sort(key=lambda x: x[0])

	start_date, end_date = db_boundary[ifp_id]
	start_date = max(db[ifp_id][0][0].replace(hour=0, minute=0, second=0, microsecond=0), start_date)

	# remove forecasts made after end_date
	db[ifp_id] = [x for x in db[ifp_id] if x[0] <= end_date]

	forecast_dates = []
	forecast_date = start_date
	while forecast_date <= end_date:
		forecast_dates.append(forecast_date.replace(hour=23, minute=59, second=59, microsecond=999))
		forecast_date += timedelta(days=1)

	if len(forecast_dates) > 0:
		db_dates[ifp_id] = forecast_dates
	else:
		# This ifp should be deleted
		deleted_ifp.append(ifp_id)

for ifp_id in deleted_ifp:
	del db[ifp_id]

max_steps = max([len(v) for k, v in db.items()])
all_ifp = np.asarray(list(db.keys()))
kf = sklearn.model_selection.KFold(shuffle=True, n_splits=5, random_state=1)
folds = [[all_ifp[f[0]], all_ifp[f[1]]] for f in kf.split(all_ifp)]

ifp_train = folds[fold_index][0]
ifp_valid = folds[fold_index][1]

n_train = len(ifp_train)
n_valid = len(ifp_valid)

print('max_steps', max_steps)
print('n_train', n_train)
print('n_valid', n_valid)

#N_Q_DIM = 64
N_K_DIM = 64
N_V_DIM = 64
N_EMB_DIM = 64
N_POS_DIM = 1

special_symbol = {
	'padding': 0,
	'unknown': 1,
}

id_counter = Counter()
for ifp_id, forecasts in db.items():
	if ifp_id in ifp_train:
		id_counter.update([f[1] for f in forecasts])

id2index = copy.deepcopy(special_symbol)
for index, value in enumerate(id_counter.most_common()):
	id2index[value[0]] = index + len(special_symbol)

### TRAIN data
def get_data(ifp):
	n = len(ifp)
	n_forecast = sum([len(v) for k, v in db_dates.items() if k in ifp])
	inputs = np.zeros((n, max_steps, 5 + n_feature + N_POS_DIM), dtype=np.float32)
	ids = np.zeros((n, max_steps), dtype=np.int32)
	state = np.zeros((n, 300), dtype=np.float32)
	target = np.zeros((n_forecast, 5), dtype=np.float32)
	answer = np.zeros(n_forecast, dtype=np.int32)
	is_ordered = np.zeros(n_forecast, dtype=bool)
	is_4 = np.zeros(n_forecast, dtype=bool)
	is_3 = np.zeros(n_forecast, dtype=bool)
	weight = np.zeros(n_forecast, dtype=np.float32)
	seq_length_mask = np.zeros((n, max_steps, max_steps), dtype=np.float32)
	gather_index = np.zeros((n_forecast, 2), dtype=np.int32)
	num_option_ary = np.zeros(n_forecast, dtype=np.int32)
	num_option_mask = np.full((n_forecast, 5), -1e32, dtype=np.float32)
	index_map = {}

	forecast_index = 0
	for index, ifp in enumerate(ifp):
		forecasts = db[ifp]

		n_total = len(forecasts)
		for i, forecast in enumerate(forecasts):
			inputs[index, i] = forecast[4:] + [i/n_total]
			forecaster_id = id2index.get(forecast[1], 1)
			ids[index, i] = forecaster_id

		forecast_dates = db_dates[ifp]
		n_forecasts = len(forecast_dates)
		activity_dates = [x[0] for x in forecasts]

		ans, ordered = db_answer[ifp]
		target[forecast_index:forecast_index+n_forecasts, ans] = 1
		answer[forecast_index:forecast_index+n_forecasts] = ans
		is_ordered[forecast_index:forecast_index+n_forecasts] = ordered
		weight[forecast_index:forecast_index+n_forecasts] = ((1.0 / n_forecasts) / n) * n_forecast
		state[index] = db_emb[ifp]
		seq_length_mask[index, :len(forecasts), :len(forecasts)] = 1

		for i, forecast_date in enumerate(forecast_dates):
			this_index = np.searchsorted(activity_dates, forecast_date)
			if this_index == 0:
				pdb.set_trace()
				print('Unexpected. Forecast dates have been adjusted for activity dates')
			gather_index[forecast_index+i, :] = [index, this_index-1]

		num_options = forecasts[0][3]
		if num_options == 4:
			is_4[forecast_index:forecast_index+n_forecasts] = True
		else:
			if num_options == 3:
				is_3[forecast_index:forecast_index+n_forecasts] = True

		num_option_ary[forecast_index:forecast_index+n_forecasts] = num_options
		num_option_mask[forecast_index:forecast_index+n_forecasts, :num_options] = 0

		index_map[ifp] = list(range(forecast_index, forecast_index+n_forecasts))
		forecast_index += n_forecasts

	inputs[np.isnan(inputs)] = 0
	input_ds = tf.data.Dataset.from_tensors(inputs)
	id_ds = tf.data.Dataset.from_tensors(ids)
	target_ds = tf.data.Dataset.from_tensors(target)
	is_ordered_ds = tf.data.Dataset.from_tensors(is_ordered)
	is_4_ds = tf.data.Dataset.from_tensors(is_4)
	is_3_ds = tf.data.Dataset.from_tensors(is_3)
	weight_ds = tf.data.Dataset.from_tensors(weight)
	seq_length_mask_ds = tf.data.Dataset.from_tensors(seq_length_mask)
	gather_index_ds = tf.data.Dataset.from_tensors(gather_index)
	num_option_mask_ds = tf.data.Dataset.from_tensors(num_option_mask)
	state_ds = tf.data.Dataset.from_tensors(state)

	all_iter = tf.data.Dataset.zip((input_ds, id_ds, target_ds, is_ordered_ds, is_4_ds, is_3_ds, weight_ds, seq_length_mask_ds, gather_index_ds, num_option_mask_ds, state_ds)).repeat(-1).make_one_shot_iterator()
	return all_iter, num_option_ary, index_map, answer, is_ordered

all_iter_train, num_option_ary_train, index_map_train, answer_train, is_ordered_train = get_data(ifp_train)
all_iter_valid, num_option_ary_valid, index_map_valid, answer_valid, is_ordered_valid = get_data(ifp_valid)

# Network placeholder
is_training = tf.placeholder_with_default(False, shape=(), name='is_training')
input_keep_prob = tf.cond(is_training, lambda:tf.constant(0.9), lambda:tf.constant(1.0))
output_keep_prob = tf.cond(is_training, lambda:tf.constant(0.9), lambda:tf.constant(1.0))
state_keep_prob = tf.cond(is_training, lambda:tf.constant(0.9), lambda:tf.constant(1.0))

all_handle = tf.placeholder(tf.string, shape=[])
[input_ph, id_ph, target_ph, is_ordered_ph, is_4_ph, is_3_ph, weight_ph, seq_length_mask_ph, gather_index_ph, num_option_mask_ph, state_ph] = \
	tf.data.Iterator.from_string_handle(all_handle, (tf.float32, tf.int32, tf.float32, tf.bool, tf.bool, tf.bool, tf.float32, tf.float32, tf.int32, tf.float32, tf.float32), \
	(tf.TensorShape([None, max_steps, 5 + n_feature + N_POS_DIM]), tf.TensorShape([None, max_steps]), tf.TensorShape([None, 5]), tf.TensorShape([None]), tf.TensorShape([None]), tf.TensorShape([None]), tf.TensorShape([None]), tf.TensorShape([None, max_steps, max_steps]), tf.TensorShape([None, 2]), tf.TensorShape([None, 5]), tf.TensorShape([None, 300]))).get_next()

#[id_ph] = tf.data.Iterator.from_string_handle(id_handle, (tf.int32, ), (tf.TensorShape([None, max_steps]), )).get_next()
#[target_ph] = tf.data.Iterator.from_string_handle(target_handle, (tf.float32, ), (tf.TensorShape([None, 5]), )).get_next()
#[is_ordered_ph] = tf.data.Iterator.from_string_handle(is_ordered_handle, (tf.bool, ), (tf.TensorShape([None]), )).get_next()
#[is_4_ph] = tf.data.Iterator.from_string_handle(is_4_handle, (tf.bool, ), (tf.TensorShape([None]), )).get_next()
#[is_3_ph] = tf.data.Iterator.from_string_handle(is_3_handle, (tf.bool, ), (tf.TensorShape([None]), )).get_next()
#[weight_ph] = tf.data.Iterator.from_string_handle(weight_handle, (tf.float32, ), (tf.TensorShape([None]), )).get_next()
#[seq_length_mask_ph] = tf.data.Iterator.from_string_handle(seq_length_mask_handle, (tf.float32, ), (tf.TensorShape([None, max_steps, max_steps]), )).get_next()
#[gather_index_ph] = tf.data.Iterator.from_string_handle(gather_index_handle, (tf.int32, ), (tf.TensorShape([None, 2]), )).get_next()
#[num_option_mask_ph] = tf.data.Iterator.from_string_handle(num_option_mask_handle, (tf.float32, ), (tf.TensorShape([None, 5]), )).get_next()


N_EMB_DIM = 300 - (5+n_feature+N_POS_DIM)
embedding = tf.get_variable('embedding', shape=(len(id2index), N_EMB_DIM), initializer=tf.random_uniform_initializer(-math.sqrt(3), math.sqrt(3)))
embedded_features = tf.nn.embedding_lookup(embedding, id_ph)
combined_input = tf.nn.dropout(tf.concat([input_ph, embedded_features], 2), keep_prob=input_keep_prob)

N_INPUT_DIM = 5+N_EMB_DIM+n_feature+N_POS_DIM
k_filter = tf.get_variable('k_w', shape=(1, N_INPUT_DIM, N_K_DIM))
k_output = tf.nn.conv1d(combined_input, k_filter, stride=1, padding='VALID')
v_filter = tf.get_variable('v_w', shape=(1, N_INPUT_DIM, N_V_DIM))
v_output = tf.nn.conv1d(combined_input, v_filter, stride=1, padding='VALID')

W_emb = tf.get_variable('embedding_weight', shape=(300, N_K_DIM), initializer=tf.glorot_uniform_initializer())
b_emb = tf.get_variable('embedding_bias', shape=(1, N_K_DIM), initializer=tf.zeros_initializer())
zero_state = tf.expand_dims(tf.matmul(state_ph, W_emb) + b_emb, 1)

#pdb.set_trace()

n_sample = tf.shape(zero_state)[0]
att_scores = tf.linalg.band_part(tf.broadcast_to(tf.expand_dims(tf.reduce_sum(k_output * zero_state, axis=2) / (N_K_DIM ** 0.5), 2), [n_sample, max_steps, max_steps]), -1, 0) * seq_length_mask_ph
att_state_series = tf.nn.dropout(tf.matmul(att_scores, v_output), keep_prob=state_keep_prob)

'''
k_filter2 = tf.get_variable('k_w2', shape=(1, N_K_DIM, N_K_DIM))
k_output2 = tf.nn.conv1d(att_state_series, k_filter2, stride=1, padding='VALID')
v_filter2 = tf.get_variable('v_w2', shape=(1, N_K_DIM, N_V_DIM))
v_output2 = tf.nn.conv1d(att_state_series, v_filter2, stride=1, padding='VALID')

att_scores2 = tf.linalg.band_part(tf.broadcast_to(tf.expand_dims(tf.reduce_sum(k_output2 * zero_state, axis=2) / (N_K_DIM ** 0.5), 2), [n_sample, max_steps, max_steps]), -1, 0) * seq_length_mask_ph
att_state_series2 = tf.nn.dropout(tf.matmul(att_scores, v_output), keep_prob=output_keep_prob)

#att_state_series2 =tf.nn.dropout(tf.matmul(att_scores2, v_output2), keep_prob=output_keep_prob)

'''
needed_state = tf.gather_nd(att_state_series, gather_index_ph)
#combined_state = tf.concat([needed_state, zero_state], 1)

#pdb.set_trace()
W1 = tf.get_variable('weight1', shape=(N_K_DIM, 5), initializer=tf.glorot_uniform_initializer())
b1 = tf.get_variable('bias1', shape=(1, 5), initializer=tf.zeros_initializer())
prediction = tf.matmul(needed_state, W1) + b1
#prediction = needed_state
#pdb.set_trace()
prob = tf.nn.softmax(prediction + num_option_mask_ph)

#prob = needed_state
#pdb.set_trace()
loss_mse = tf.math.reduce_sum(tf.math.squared_difference(target_ph, prob), axis=1)

prob_1 = tf.stack([tf.reduce_sum(tf.gather(prob, [0], axis=1), axis=1), tf.reduce_sum(tf.gather(prob, [1, 2, 3, 4], axis=1), axis=1)], axis=1)
true_1 = tf.stack([tf.reduce_sum(tf.gather(target_ph, [0], axis=1), axis=1), tf.reduce_sum(tf.gather(target_ph, [1, 2, 3, 4], axis=1), axis=1)], axis=1)
loss_1 = tf.math.reduce_sum(tf.math.squared_difference(prob_1, true_1), axis=1)

prob_2 = tf.stack([tf.reduce_sum(tf.gather(prob, [0, 1], axis=1), axis=1), tf.reduce_sum(tf.gather(prob, [2, 3, 4], axis=1), axis=1)], axis=1)
true_2 = tf.stack([tf.reduce_sum(tf.gather(target_ph, [0, 1], axis=1), axis=1), tf.reduce_sum(tf.gather(target_ph, [2, 3, 4], axis=1), axis=1)], axis=1)
loss_2 = tf.math.reduce_sum(tf.math.squared_difference(prob_2, true_2), axis=1)

prob_3 = tf.stack([tf.reduce_sum(tf.gather(prob, [0, 1, 2], axis=1), axis=1), tf.reduce_sum(tf.gather(prob, [3, 4], axis=1), axis=1)], axis=1)
true_3 = tf.stack([tf.reduce_sum(tf.gather(target_ph, [0, 1, 2], axis=1), axis=1), tf.reduce_sum(tf.gather(target_ph, [3, 4], axis=1), axis=1)], axis=1)
loss_3 = tf.math.reduce_sum(tf.math.squared_difference(prob_3, true_3), axis=1)

prob_4 = tf.stack([tf.reduce_sum(tf.gather(prob, [0, 1, 2, 3], axis=1), axis=1), tf.reduce_sum(tf.gather(prob, [4], axis=1), axis=1)], axis=1)
true_4 = tf.stack([tf.reduce_sum(tf.gather(target_ph, [0, 1, 2, 3], axis=1), axis=1), tf.reduce_sum(tf.gather(target_ph, [4], axis=1), axis=1)], axis=1)
loss_4 = tf.math.reduce_sum(tf.math.squared_difference(prob_4, true_4), axis=1)

loss_brier = tf.math.reduce_mean(tf.stack([loss_1, loss_2, loss_3, loss_4], axis=1), axis=1)


prob_4_1 = tf.stack([tf.reduce_sum(tf.gather(prob, [0], axis=1), axis=1), tf.reduce_sum(tf.gather(prob, [1, 2, 3], axis=1), axis=1)], axis=1)
true_4_1 = tf.stack([tf.reduce_sum(tf.gather(target_ph, [0], axis=1), axis=1), tf.reduce_sum(tf.gather(target_ph, [1, 2, 3], axis=1), axis=1)], axis=1)
loss_4_1 = tf.math.reduce_sum(tf.math.squared_difference(prob_4_1, true_4_1), axis=1)

prob_4_2 = tf.stack([tf.reduce_sum(tf.gather(prob, [0, 1], axis=1), axis=1), tf.reduce_sum(tf.gather(prob, [2, 3], axis=1), axis=1)], axis=1)
true_4_2 = tf.stack([tf.reduce_sum(tf.gather(target_ph, [0, 1], axis=1), axis=1), tf.reduce_sum(tf.gather(target_ph, [2, 3], axis=1), axis=1)], axis=1)
loss_4_2 = tf.math.reduce_sum(tf.math.squared_difference(prob_4_2, true_4_2), axis=1)

prob_4_3 = tf.stack([tf.reduce_sum(tf.gather(prob, [0, 1, 2], axis=1), axis=1), tf.reduce_sum(tf.gather(prob, [3], axis=1), axis=1)], axis=1)
true_4_3 = tf.stack([tf.reduce_sum(tf.gather(target_ph, [0, 1, 2], axis=1), axis=1), tf.reduce_sum(tf.gather(target_ph, [3], axis=1), axis=1)], axis=1)
loss_4_3 = tf.math.reduce_sum(tf.math.squared_difference(prob_4_3, true_4_3), axis=1)

loss_brier_4 = tf.math.reduce_mean(tf.stack([loss_4_1, loss_4_2, loss_4_3], axis=1), axis=1)

prob_3_1 = tf.stack([tf.reduce_sum(tf.gather(prob, [0], axis=1), axis=1), tf.reduce_sum(tf.gather(prob, [1, 2], axis=1), axis=1)], axis=1)
true_3_1 = tf.stack([tf.reduce_sum(tf.gather(target_ph, [0], axis=1), axis=1), tf.reduce_sum(tf.gather(target_ph, [1, 2], axis=1), axis=1)], axis=1)
loss_3_1 = tf.math.reduce_sum(tf.math.squared_difference(prob_3_1, true_3_1), axis=1)

prob_3_2 = tf.stack([tf.reduce_sum(tf.gather(prob, [0, 1], axis=1), axis=1), tf.reduce_sum(tf.gather(prob, [2], axis=1), axis=1)], axis=1)
true_3_2 = tf.stack([tf.reduce_sum(tf.gather(target_ph, [0, 1], axis=1), axis=1), tf.reduce_sum(tf.gather(target_ph, [2], axis=1), axis=1)], axis=1)
loss_3_2 = tf.math.reduce_sum(tf.math.squared_difference(prob_3_2, true_3_2), axis=1)

loss_brier_3 = tf.math.reduce_mean(tf.stack([loss_3_1, loss_3_2], axis=1), axis=1)

loss_combined = tf.where(is_ordered_ph, tf.where(is_4_ph, loss_brier_4, tf.where(is_3_ph, loss_brier_3, loss_brier)), loss_mse)
loss_weighted = tf.losses.compute_weighted_loss(loss_combined, weight_ph)


loss_weighted_reg = loss_weighted
variables = [v for v in tf.trainable_variables()]

for v in variables:
	loss_weighted_reg += 1e-8 * tf.nn.l2_loss(v)# + 1e-8 * tf.losses.absolute_difference(v, tf.zeros(tf.shape(v)))

lr = tf.Variable(0.1, trainable=False)
lr_decay_op = lr.assign(lr * 0.95)
optimizer = tf.train.AdamOptimizer(lr)

gradients, variables = zip(*optimizer.compute_gradients(loss_weighted_reg))
#gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
train_op = optimizer.apply_gradients(zip(gradients, variables))

saver = tf.train.Saver()
save_dir = '{}/{}/{}'.format(model_name, '_'.join(feature_used).replace('/', '_').replace(' ', '_'), fold_index)
os.makedirs(save_dir, exist_ok=True)
save_path = save_dir + '/model.ckpt'

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	all_handle_train = sess.run(all_iter_train.string_handle())
	all_handle_valid = sess.run(all_iter_valid.string_handle())

	#pdb.set_trace()
	valid_scores = []
	smallest_loss = float('inf')
	smallest_train_loss = float('inf')

	wait = 0
	n_lr_decay = 5
	n_reset_weight = 200
	smallest_weight = None

	def _save_weight():
		global smallest_weight
		tf_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
		smallest_weight = sess.run(tf_vars)

	def _load_weights():
		tf_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
		ops = []
		for i_tf in range(len(tf_vars)):
			ops.append(tf.assign(tf_vars[i_tf], smallest_weight[i_tf]))
		sess.run(ops)

	for i in range(200):
		train_loss, train_pred, _train_step = sess.run(
			[loss_weighted, prob, train_op],
				feed_dict={
					all_handle: all_handle_train,
					is_training: True
				}
		)
		#pdb.set_trace()

		valid_loss, valid_pred = sess.run(
			[loss_weighted, prob],
				feed_dict={
					all_handle: all_handle_valid,
					is_training: False
				}
		)

		valid_scores.append(valid_loss)
		print('Epoch: {}, train loss: {}, valid loss: {}, min valid loss so far: {}'.format(i, train_loss, valid_loss, np.min(valid_scores)))

		if valid_loss < smallest_loss:
			smallest_loss = valid_loss
			smallest_train_loss = train_loss
			_save_weight()
			wait = 0
			print('New smallest')
		else:
			wait += 1
			print('Wait {}'.format(wait))
			if wait % n_lr_decay == 0:
				sess.run(lr_decay_op)
				print('Apply lr decay, new lr: %f' % lr.eval())

			if wait % n_reset_weight == 0:
				_load_weights()
				wait = 0
				print('Reset weights')

		# for verification purpose only
		if i == 0:
			train_briers = np.asarray([brier(p[:num_option_ary_train[i]], answer_train[i], is_ordered_train[i]) for i, p in enumerate(train_pred)])
			valid_briers = np.asarray([brier(p[:num_option_ary_valid[i]], answer_valid[i], is_ordered_valid[i]) for i, p in enumerate(valid_pred)])

			db_brier_train = {}
			for ifp in ifp_train:
				index = index_map_train[ifp]
				scores = train_briers[index]
				db_brier_train[ifp] = np.mean(scores)

			db_brier_valid = {}
			for ifp in ifp_valid:
				index = index_map_valid[ifp]
				scores = valid_briers[index]
				db_brier_valid[ifp] = np.mean(scores)

			print(i, train_loss, valid_loss, np.mean(list(db_brier_train.values())), np.mean(list(db_brier_valid.values())))

	_load_weights()

	valid_loss, valid_pred = sess.run(
		[loss_weighted, prob],
			feed_dict={
				all_handle: all_handle_valid,
				is_training: False
			}
	)

	print('final test loss', valid_loss)

	with open(save_path.replace('.ckpt', '.pickle'), 'wb') as fout:
		pickle.dump(smallest_weight, fout, pickle.HIGHEST_PROTOCOL)

	saver.save(sess, save_path)
	print('Before exit')


print('OK')
