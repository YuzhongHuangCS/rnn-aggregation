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
from collections import OrderedDict, Counter, defaultdict
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
from nltk.tokenize import word_tokenize

word_embedding = {}
if os.path.exists('deps.words.pickle'):
	print('Loading embedding from cache')
	with open('deps.words.pickle', 'rb') as fin:
		word_embedding = pickle.load(fin)
else:
	print('Loading embedding from text')
	for line in open('deps.words.txt', encoding='utf-8'):
		parts = line.rstrip().split(' ')
		word = parts[0]
		embs = [float(x) for x in parts[1:]]
		word_embedding[word] = embs
	with open('deps.words.pickle', 'wb') as fout:
		pickle.dump(word_embedding, fout, pickle.HIGHEST_PROTOCOL)

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
db_emb = {}

def embedding_lookup(t):
	e = word_embedding
	return e.get(t, e.get(t.lower(), e.get(t.upper(), e.get(t.capitalize(), None))))

for filename in ('data/dump_questions_rcta.csv', 'data/dump_questions_rctb.csv', 'data/rctb/questions.csv'):
	df_question = pd.read_csv(filename)
	for index, row in df_question.iterrows():
		if row['is_resolved']:
			if filename == 'data/rctb/questions.csv':
				ifp_id = row['hfc_id']
			else:
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

					start_date = dateutil.parser.parse(row['start_date']).replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)
					end_date = dateutil.parser.parse(row['end_date']).replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)

					forecast_dates = []
					forecast_date = start_date
					while forecast_date <= end_date:
						forecast_dates.append(forecast_date.replace(hour=23, minute=59, second=59, microsecond=999))
						forecast_date += timedelta(days=1)
					db_dates[ifp_id] = forecast_dates

					features = []
					for w in word_tokenize(row['title']):
						feature = embedding_lookup(w)
						if feature is not None:
							features.append(feature)

					feature_mean = np.mean(np.asarray(features), axis=0)
					db_emb[ifp_id] = feature_mean

			except ValueError as e:
				pdb.set_trace()
				print(e)

human_feature = pd.read_csv('data/human_features.csv').drop_duplicates(subset=['date', 'ifp_id'], keep='last')
hf_cols = human_feature.columns.tolist()
hf_cols.remove('ifp_id')
hf_cols.remove('date')
hf_cols.remove('stage')
hf_cols.remove('p_updates')
hf_cols.remove('Health/Disease')
human_feature = human_feature.drop(columns=hf_cols)

ts_feature = pd.read_csv('data/ts_features.csv')

human_feature_te = pd.read_csv('data/rctb/meta_human_features.csv').drop_duplicates(subset=['date', 'ifp_id'], keep='last')[human_feature.columns]
ts_feature_te = pd.read_csv('data/rctb/meta_ts_features.csv')
tf_cols = ts_feature_te.columns.tolist()
tf_cols.remove('ifp_id')
tf_cols.remove('date')
tf_cols.remove('ratio')
ts_feature_te = ts_feature_te.drop(columns=tf_cols)

ts_feature = ts_feature[ts_feature_te.columns]
n_feature = human_feature.shape[1] + ts_feature.shape[1] - 4
n_feature = 0

human_dict = defaultdict(dict)
for h_f in (human_feature, human_feature_te):
	for index, row in h_f.iterrows():
		ifp_id = row['ifp_id']
		date = row['date']

		if date in human_dict[ifp_id]:
			pdb.set_trace()
			print('Duplicate feature')
		else:
			human_dict[ifp_id][date] = row.drop(labels=['ifp_id', 'date']).values

ts_dict = defaultdict(dict)
for t_f in (ts_feature, ts_feature_te):
	for index, row in t_f.iterrows():
		ifp_id = row['ifp_id']
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

df = pd.read_csv('data/human.csv')
#df.fillna(0, inplace=True)
#pdb.set_trace()
#np.unique(df[df['ifp_id'].isin(ifp_all)]['user_id']).shape
db = defaultdict(list)
human_ifp_rcta_set = set()

for index, row in df.iterrows():
	date = dateutil.parser.parse(row['date']).replace(tzinfo=None)
	user_id = row['user_id']
	ifp_id = row['ifp_id']

	if ifp_id not in db_answer:
		continue

	human_ifp_rcta_set.add(ifp_id)
	num_options = row['num_options']
	option_1 = row['option_1'] / 100.0
	option_2 = row['option_2'] / 100.0
	option_3 = row['option_3'] / 100.0
	option_4 = row['option_4'] / 100.0
	option_5 = row['option_5'] / 100.0

	if num_options == 1:
		num_options = 2

	#cf = get_feature(ifp_id, datetime.strftime(date, "%Y-%m-%d"))
	db[ifp_id].append([date,user_id,ifp_id,num_options,option_1,option_2,option_3,option_4,option_5])# + cf.tolist())

db_te = defaultdict(list)
human_ifp_rctb_set = set()
df_te = pd.read_csv('data/rctb/te_forecasts.csv')
for index, row in df_te.iterrows():
	date = dateutil.parser.parse(row['created_at']).replace(tzinfo=None)
	user_id = row['user_guid']
	ifp_id = row['hfc_id']

	if ifp_id not in db_answer:
		continue

	human_ifp_rctb_set.add(ifp_id)
	num_options = row['num_options']
	option_1 = row['option_1']
	option_2 = row['option_2']
	option_3 = row['option_3']
	option_4 = row['option_4']
	option_5 = row['option_5']

	if num_options == 1:
		num_options = 2

	#cf = get_feature(ifp_id, datetime.strftime(date, "%Y-%m-%d"))
	db_te[ifp_id].append([date,user_id,ifp_id,num_options,option_1,option_2,option_3,option_4,option_5])# + cf.tolist())

machine_ifp_rcta_set = set()
machine_df = pd.read_csv('data/machine_all.csv').drop_duplicates(subset=['date', 'machine_model', 'ifp_id'], keep='last')
for index, row in machine_df.iterrows():
	date = dateutil.parser.parse(row['date'])
	machine_model = row['machine_model']
	ifp_id = row['ifp_id']

	if ifp_id not in db_answer:
		continue

	if machine_model not in ('Auto ARIMA', 'M4-Meta', 'Arithmetic RW', 'DS-Holt', 'DS-Holt-damped', 'DS-RW', 'DS-SES', 'ETS', 'Geometric RW', 'M4-Comb', 'Mean', 'NNETAR', 'RW', 'RW-DRIFT', 'RW-SEAS', 'STLM-AR', 'TBATS', 'THETA'):
		continue

	machine_ifp_rcta_set.add(ifp_id)
	'''
	if machine_model == 'M4-Meta':
		date = date.replace(microsecond=1)

	if machine_model == 'Arithmetic RW':
		date = date.replace(microsecond=2)

	if machine_model == 'DS-Holt':
		date = date.replace(microsecond=3)

	if machine_model == 'DS-Holt-damped':
		date = date.replace(microsecond=4)

	if machine_model == 'DS-RW':
		date = date.replace(microsecond=5)

	if machine_model == 'DS-SES':
		date = date.replace(microsecond=6)

	if machine_model == 'ETS':
		date = date.replace(microsecond=7)

	if machine_model == 'Geometric RW':
		date = date.replace(microsecond=8)

	if machine_model == 'M4-Comb':
		date = date.replace(microsecond=9)

	if machine_model == 'Mean':
		date = date.replace(microsecond=10)

	if machine_model == 'NNETAR':
		date = date.replace(microsecond=11)

	if machine_model == 'RW':
		date = date.replace(microsecond=12)

	if machine_model == 'RW-DRIFT':
		date = date.replace(microsecond=13)

	if machine_model == 'RW-SEAS':
		date = date.replace(microsecond=14)

	if machine_model == 'STLM-AR':
		date = date.replace(microsecond=15)

	if machine_model == 'TBATS':
		date = date.replace(microsecond=16)

	if machine_model == 'THETA':
		date = date.replace(microsecond=17)


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
		db[ifp_id] = []
		#pdb.seift_trace()
		#print("Didn't expect any ifp have human forecast but don't have machine forecast")

	#cf = get_feature(ifp_id, datetime.strftime(date, "%Y-%m-%d"))
	db[ifp_id].append([date,machine_model,ifp_id,num_options,option_1,option_2,option_3,option_4,option_5])# + cf.tolist())
	'''
machine_ifp_rctb_set = set()
machine_df_te = pd.read_csv('data/rctb/machine.csv').drop_duplicates(subset=['created_at', 'machine_model', 'hfc_id'], keep='last')
for index, row in machine_df_te.iterrows():
	date = dateutil.parser.parse(row['created_at']).replace(tzinfo=None)
	machine_model = row['machine_model']
	ifp_id = row['hfc_id']

	if ifp_id not in db_answer:
		continue

	if machine_model not in ('Auto ARIMA', 'M4-Meta', 'Arithmetic RW', 'DS-Holt', 'DS-Holt-damped', 'DS-RW', 'DS-SES', 'ETS', 'Geometric RW', 'M4-Comb', 'Mean', 'NNETAR', 'RW', 'RW-DRIFT', 'RW-SEAS', 'STLM-AR', 'TBATS', 'THETA'):
		continue

	machine_ifp_rctb_set.add(ifp_id)
	'''
	if machine_model == 'M4-Meta':
		date = date.replace(microsecond=1)

	if machine_model == 'Arithmetic RW':
		date = date.replace(microsecond=2)

	if machine_model == 'DS-Holt':
		date = date.replace(microsecond=3)

	if machine_model == 'DS-Holt-damped':
		date = date.replace(microsecond=4)

	if machine_model == 'DS-RW':
		date = date.replace(microsecond=5)

	if machine_model == 'DS-SES':
		date = date.replace(microsecond=6)

	if machine_model == 'ETS':
		date = date.replace(microsecond=7)

	if machine_model == 'Geometric RW':
		date = date.replace(microsecond=8)

	if machine_model == 'M4-Comb':
		date = date.replace(microsecond=9)

	if machine_model == 'Mean':
		date = date.replace(microsecond=10)

	if machine_model == 'NNETAR':
		date = date.replace(microsecond=11)

	if machine_model == 'RW':
		date = date.replace(microsecond=12)

	if machine_model == 'RW-DRIFT':
		date = date.replace(microsecond=13)

	if machine_model == 'RW-SEAS':
		date = date.replace(microsecond=14)

	if machine_model == 'STLM-AR':
		date = date.replace(microsecond=15)

	if machine_model == 'TBATS':
		date = date.replace(microsecond=16)

	if machine_model == 'THETA':
		date = date.replace(microsecond=17)


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

	if ifp_id not in db_te:
		db_te[ifp_id] = []
		#pdb.set_trace()
		#print("Didn't expect any ifp have human forecast but don't have machine forecast")

	#cf = get_feature(ifp_id, datetime.strftime(date, "%Y-%m-%d"))
	db_te[ifp_id].append([date,machine_model,ifp_id,num_options,option_1,option_2,option_3,option_4,option_5])# + cf.tolist())
	'''

for ifp_id in db:
	db[ifp_id].sort(key=lambda x: x[0])

for ifp_id in db_te:
	db_te[ifp_id].sort(key=lambda x: x[0])

max_steps = max([len(v) for k, v in db.items()] + [len(v) for k, v in db_te.items()])

intersect_ifp_rcta = human_ifp_rcta_set.intersection(machine_ifp_rcta_set)
intersect_ifp_rctb = human_ifp_rctb_set.intersection(machine_ifp_rctb_set)
all_ifp = np.asarray(list(intersect_ifp_rcta))

kf = sklearn.model_selection.KFold(shuffle=True, n_splits=5, random_state=2019)
folds = [[all_ifp[f[0]], all_ifp[f[1]]] for f in kf.split(all_ifp)]
fold_index = 0

#ifp_train = folds[fold_index][0]
#ifp_valid = folds[fold_index][1]
ifp_train = all_ifp
ifp_valid = np.asarray(list(intersect_ifp_rctb))
ifp_test = np.asarray(list(intersect_ifp_rctb))

n_train = len(ifp_train)
n_valid = len(ifp_valid)
n_test = len(ifp_test)

N_RNN_DIM = 32
N_EMB_DIM = 4

special_symbol = {
	'padding': 0,
	'unknown': 1,
	'Auto ARIMA': 2,
	'M4-Meta': 3,
	'Arithmetic RW': 4,
	'DS-Holt': 5,
	'DS-Holt-damped': 6,
	'DS-RW': 7,
	'DS-SES': 8,
	'ETS': 9,
	'Geometric RW': 10,
	'M4-Comb': 11,
	'Mean': 12,
	'NNETAR': 13,
	'RW': 14,
	'RW-DRIFT': 15,
	'RW-SEAS': 16,
	'STLM-AR': 17,
	'TBATS': 18,
	'THETA': 19,
}

#id_counter = Counter()
#id_counter.update(df[df['ifp_id'].isin(ifp_train)]['user_id'])
id2index = copy.deepcopy(special_symbol)

#for index, value in enumerate(id_counter.most_common()):
#	id2index[value[0]] = index + len(special_symbol)

### TRAIN data
n_forecast_train = sum([len(v) for k, v in db_dates.items() if k in ifp_train])

input_train = np.zeros((n_train, max_steps, 5 + n_feature))
id_train = np.zeros((n_train, max_steps, 1), dtype=int)
state_train = np.zeros((n_train, 300))
target_train = np.zeros((n_forecast_train, 5))
answer_train = np.zeros(n_forecast_train, dtype=int)
is_ordered_train = np.zeros(n_forecast_train, dtype=bool)
is_4_train = np.zeros(n_forecast_train, dtype=bool)
is_3_train = np.zeros(n_forecast_train, dtype=bool)
weight_train = np.zeros(n_forecast_train)
seq_length_train = np.zeros(n_train, dtype=int)
gather_index_train = np.zeros((n_forecast_train, 2), dtype=int)
num_option_ary_train = np.zeros(n_forecast_train, dtype=int)
num_option_mask_train = np.full((n_forecast_train, 5), -1e32)
index_map_train = {}

forecast_index = 0
for index, ifp in enumerate(ifp_train):
	forecasts = db[ifp]

	for i, forecast in enumerate(forecasts):
		input_train[index, i] = forecast[4:]
		forecaster_id = id2index.get(forecast[1], 1)
		id_train[index, i] = forecaster_id

	state_train[index] = db_emb[ifp]
	forecast_dates = db_dates[ifp]
	n_forecasts = len(forecast_dates)
	activity_dates = [x[0] for x in forecasts]

	answer, is_ordered = db_answer[ifp]
	target_train[forecast_index:forecast_index+n_forecasts, answer] = 1
	answer_train[forecast_index:forecast_index+n_forecasts] = answer
	is_ordered_train[forecast_index:forecast_index+n_forecasts] = is_ordered
	weight_train[forecast_index:forecast_index+n_forecasts] = ((1.0 / n_forecasts) / n_train) * n_forecast_train
	seq_length_train[index] = len(forecasts)

	for i, forecast_date in enumerate(forecast_dates):
		this_index = np.searchsorted(activity_dates, forecast_date)
		# hack here! If no forecast on the first day, use the first subsequent
		if this_index == 0:
			this_index = 1
		gather_index_train[forecast_index+i, :] = [index, this_index-1]

	num_options = forecasts[0][3]
	if num_options == 4:
		is_4_train[forecast_index:forecast_index+n_forecasts] = True
	else:
		if num_options == 3:
			is_3_train[forecast_index:forecast_index+n_forecasts] = True

	num_option_ary_train[forecast_index:forecast_index+n_forecasts] = num_options
	num_option_mask_train[forecast_index:forecast_index+n_forecasts, :num_options] = 0

	index_map_train[ifp] = list(range(forecast_index, forecast_index+n_forecasts))
	forecast_index += n_forecasts

input_train[np.isnan(input_train)] = 0

### VALID data
n_forecast_valid = sum([len(v) for k, v in db_dates.items() if k in ifp_valid])

input_valid = np.zeros((n_valid, max_steps, 5 + n_feature))
id_valid = np.zeros((n_valid, max_steps, 1), dtype=int)
state_valid = np.zeros((n_valid, 300))
target_valid = np.zeros((n_forecast_valid, 5))
answer_valid = np.zeros(n_forecast_valid, dtype=int)
is_ordered_valid = np.zeros(n_forecast_valid, dtype=bool)
is_4_valid = np.zeros(n_forecast_valid, dtype=bool)
is_3_valid = np.zeros(n_forecast_valid, dtype=bool)
weight_valid = np.zeros(n_forecast_valid)
seq_length_valid = np.zeros(n_valid, dtype=int)
gather_index_valid = np.zeros((n_forecast_valid, 2), dtype=int)
num_option_ary_valid = np.zeros(n_forecast_valid, dtype=int)
num_option_mask_valid = np.full((n_forecast_valid, 5), -1e32)
index_map_valid = {}

forecast_index = 0
for index, ifp in enumerate(ifp_valid):
	forecasts = db_te[ifp]

	for i, forecast in enumerate(forecasts):
		input_valid[index, i] = forecast[4:]
		forecaster_id = id2index.get(forecast[1], 1)
		id_valid[index, i] = forecaster_id

	state_valid[index] = db_emb[ifp]
	forecast_dates = db_dates[ifp]
	n_forecasts = len(forecast_dates)
	activity_dates = [x[0] for x in forecasts]

	answer, is_ordered = db_answer[ifp]
	target_valid[forecast_index:forecast_index+n_forecasts, answer] = 1
	answer_valid[forecast_index:forecast_index+n_forecasts] = answer
	is_ordered_valid[forecast_index:forecast_index+n_forecasts] = is_ordered
	weight_valid[forecast_index:forecast_index+n_forecasts] = ((1.0 / n_forecasts) / n_valid) * n_forecast_valid
	seq_length_valid[index] = len(forecasts)

	for i, forecast_date in enumerate(forecast_dates):
		this_index = np.searchsorted(activity_dates, forecast_date)
		# hack here! If no forecast on the first day, use the first subsequent
		if this_index == 0:
			this_index = 1
		gather_index_valid[forecast_index+i, :] = [index, this_index-1]

	num_options = forecasts[0][3]
	if num_options == 4:
		is_4_valid[forecast_index:forecast_index+n_forecasts] = True
	else:
		if num_options == 3:
			is_3_valid[forecast_index:forecast_index+n_forecasts] = True

	num_option_ary_valid[forecast_index:forecast_index+n_forecasts] = num_options
	num_option_mask_valid[forecast_index:forecast_index+n_forecasts, :num_options] = 0

	index_map_valid[ifp] = list(range(forecast_index, forecast_index+n_forecasts))
	forecast_index += n_forecasts

input_valid[np.isnan(input_valid)] = 0

### TEST data
n_forecast_test = sum([len(v) for k, v in db_dates.items() if k in ifp_test])

input_test = np.zeros((n_test, max_steps, 5 + n_feature))
id_test = np.zeros((n_test, max_steps, 1), dtype=int)
state_test = np.zeros((n_test, 300))
target_test = np.zeros((n_forecast_test, 5))
answer_test = np.zeros(n_forecast_test, dtype=int)
is_ordered_test = np.zeros(n_forecast_test, dtype=bool)
is_4_test = np.zeros(n_forecast_test, dtype=bool)
is_3_test = np.zeros(n_forecast_test, dtype=bool)
weight_test = np.zeros(n_forecast_test)
seq_length_test = np.zeros(n_test, dtype=int)
gather_index_test = np.zeros((n_forecast_test, 2), dtype=int)
num_option_ary_test = np.zeros(n_forecast_test, dtype=int)
num_option_mask_test = np.full((n_forecast_test, 5), -1e32)
index_map_test = {}

forecast_index = 0
for index, ifp in enumerate(ifp_test):
	forecasts = db_te[ifp]

	for i, forecast in enumerate(forecasts):
		input_test[index, i] = forecast[4:]

		forecaster_id = id2index.get(forecast[1], 1)
		id_test[index, i] = forecaster_id

	state_test[index] = db_emb[ifp]
	forecast_dates = db_dates[ifp]
	n_forecasts = len(forecast_dates)
	activity_dates = [x[0] for x in forecasts]

	answer, is_ordered = db_answer[ifp]
	target_test[forecast_index:forecast_index+n_forecasts, answer] = 1
	answer_test[forecast_index:forecast_index+n_forecasts] = answer
	is_ordered_test[forecast_index:forecast_index+n_forecasts] = is_ordered
	weight_test[forecast_index:forecast_index+n_forecasts] = ((1.0 / n_forecasts) / n_test) * n_forecast_test
	seq_length_test[index] = len(forecasts)

	for i, forecast_date in enumerate(forecast_dates):
		this_index = np.searchsorted(activity_dates, forecast_date)
		# hack here! If no forecast on the first day, use the first subsequent
		if this_index == 0:
			this_index = 1
		gather_index_test[forecast_index+i, :] = [index, this_index-1]

	num_options = forecasts[0][3]
	if num_options == 4:
		is_4_test[forecast_index:forecast_index+n_forecasts] = True
	else:
		if num_options == 3:
			is_3_test[forecast_index:forecast_index+n_forecasts] = True

	num_option_ary_test[forecast_index:forecast_index+n_forecasts] = num_options
	num_option_mask_test[forecast_index:forecast_index+n_forecasts, :num_options] = 0

	index_map_test[ifp] = list(range(forecast_index, forecast_index+n_forecasts))
	forecast_index += n_forecasts

input_test[np.isnan(input_test)] = 0

# Network placeholder
is_training = tf.placeholder_with_default(False, shape=(), name='is_training')
input_placeholder = tf.placeholder(tf.float32, [None, max_steps, 5 + n_feature])
id_placeholder = tf.placeholder(tf.int32, [None, max_steps, 1])
target_placeholder = tf.placeholder(tf.float32, [None, 5])
is_ordered_placeholder = tf.placeholder(tf.bool, [None])
is_4_placeholder = tf.placeholder(tf.bool, [None])
is_3_placeholder = tf.placeholder(tf.bool, [None])
weight_placeholder = tf.placeholder(tf.float32, [None])
seq_length_placeholder = tf.placeholder(tf.int32, [None])
gather_index_placeholder = tf.placeholder(tf.int32, [None, 2])
num_option_mask_placeholder = tf.placeholder(tf.float32, [None, 5])

embedding = tf.get_variable('embedding', shape=(len(id2index), N_EMB_DIM), initializer=tf.random_uniform_initializer(-math.sqrt(3), math.sqrt(3)))
embedded_features = tf.nn.embedding_lookup(embedding, id_placeholder)

combined_input = input_placeholder
#combined_input = tf.concat([input_placeholder, embedded_features[:, :, 0, :]], 2)

cell = tf.nn.rnn_cell.GRUCell(N_RNN_DIM, kernel_initializer=tf.orthogonal_initializer(), bias_initializer=tf.zeros_initializer())
input_keep_prob = tf.cond(is_training, lambda:tf.constant(0.95), lambda:tf.constant(1.0))
output_keep_prob = tf.cond(is_training, lambda:tf.constant(0.95), lambda:tf.constant(1.0))
state_keep_prob = tf.cond(is_training, lambda:tf.constant(0.95), lambda:tf.constant(1.0))
cell_dropout = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob = input_keep_prob, output_keep_prob = output_keep_prob, state_keep_prob = state_keep_prob)
initial_state = tf.placeholder(tf.float32, [None, 300])
W_emb = tf.get_variable('embedding_weight', shape=(300, N_RNN_DIM))
b_emb = tf.get_variable('embedding_bias', shape=(1, N_RNN_DIM))
zero_state = tf.matmul(initial_state, W_emb) + b_emb

state_series, _ = tf.nn.dynamic_rnn(cell_dropout, combined_input, sequence_length=seq_length_placeholder, initial_state=zero_state)

W1 = tf.get_variable('weight1', shape=(N_RNN_DIM, 5), initializer=tf.glorot_uniform_initializer())
b1 = tf.get_variable('bias1', shape=(1, 5), initializer=tf.zeros_initializer())
needed_state = tf.gather_nd(state_series, gather_index_placeholder)
prediction = tf.matmul(tf.nn.tanh(needed_state), W1) + b1
prob = tf.nn.softmax(tf.math.add(prediction, num_option_mask_placeholder))
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
	loss_weighted_reg += 1e-4 * tf.nn.l2_loss(v)# + 0.001 * tf.losses.absolute_difference(v, tf.zeros(tf.shape(v)))

lr = tf.Variable(0.01, trainable=False)
lr_decay_op = lr.assign(lr * 0.95)
optimizer = tf.train.AdamOptimizer(lr)

gradients, variables = zip(*optimizer.compute_gradients(loss_weighted_reg))
gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
train_op = optimizer.apply_gradients(zip(gradients, variables))

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	#pdb.set_trace()
	valid_scores = []

	smallest_loss = float('inf')
	wait = 0
	#n_patience = 20
	n_lr_decay = 5
	n_reset_weight = 20
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
					num_option_mask_placeholder: num_option_mask_train,
					initial_state: state_train,
					is_training: True
				}
		)

		valid_loss, valid_pred = sess.run(
			[loss_weighted, prob],
				feed_dict={
					input_placeholder: input_valid,
					id_placeholder: id_valid,
					target_placeholder: target_valid,
					is_ordered_placeholder: is_ordered_valid,
					is_4_placeholder: is_4_valid,
					is_3_placeholder: is_3_valid,
					weight_placeholder: weight_valid,
					seq_length_placeholder: seq_length_valid,
					gather_index_placeholder: gather_index_valid,
					num_option_mask_placeholder: num_option_mask_valid,
					initial_state: state_valid,
					is_training: False
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
					num_option_mask_placeholder: num_option_mask_test,
					initial_state: state_test,
					is_training: False
				}
		)

		valid_scores.append(valid_loss)
		print('Epoch: {}, train loss: {}, valid loss: {}, min valid loss so far: {}, test loss: {}'.format(i, train_loss, valid_loss, np.min(valid_scores), test_loss))

		if valid_loss < smallest_loss:
			smallest_loss = valid_loss
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
			test_briers = np.asarray([brier(p[:num_option_ary_test[i]], answer_test[i], is_ordered_test[i]) for i, p in enumerate(test_pred)])

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

			db_brier_test = {}
			for ifp in ifp_test:
				index = index_map_test[ifp]
				scores = test_briers[index]
				db_brier_test[ifp] = np.mean(scores)

			print(i, train_loss, valid_loss, test_loss, np.mean(list(db_brier_train.values())), np.mean(list(db_brier_valid.values())), np.mean(list(db_brier_test.values())))

	_load_weights()
	print('min valid loss', np.min(valid_scores))
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
				num_option_mask_placeholder: num_option_mask_test,
				initial_state: state_test,
				is_training: False
			}
	)

	def save_model(filename):
		with open(filename, 'wb') as fout:
			pickle.dump(smallest_weight, fout, pickle.HIGHEST_PROTOCOL)

	print('test loss', test_loss)
	pdb.set_trace()
	print('Before exit')

print('OK')