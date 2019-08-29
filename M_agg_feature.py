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
from collections import OrderedDict, Counter,defaultdict
import pickle
import sklearn.model_selection
import sklearn.decomposition
import sklearn.cluster
import scipy.stats
# import tensorflow as tf
import torch

import random
import copy
from briercompute import brier, get_user_brier
from datetime import datetime, timedelta
import math

N_RNN_DIM = 32
N_EMB_DIM = 8
N_HIDDEN_DIM = 16
EPS = 1e-8

def is_ordered(opt):
	keywords = ['Less', 'Between', 'More', 'inclusive','less', 'between', 'more']
	if len(opt) <= 2: #binary
		return False
	for o in opt:
		if any(x in o for x in keywords):
			return True

	return False

if not os.path.exists('cache.ckpt'):
	print('Reading data')
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

						start_date = dateutil.parser.parse(row['start_date']).replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)
						end_date = dateutil.parser.parse(row['end_date']).replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)

						forecast_dates = []
						forecast_date = start_date
						while forecast_date <= end_date:
							forecast_dates.append(forecast_date.replace(hour=23, minute=59, second=59, microsecond=999))
							forecast_date += timedelta(days=1)
						db_dates[ifp_id] = forecast_dates
				except ValueError as e:
					pdb.set_trace()
					print(e)

	human_feature_list = ['Health/Disease', 'Macroeconomics/Finance', 'Natural Sciences/Climate',
	'Other', 'Politics/Intl Relations', 'Technology', 'entropy_b',
	'entropy_c', 'entropy_d', 'entropy_human', 'entropy_sage', 'entropy_te',
	'n_forecasts', 'n_forecasts_b', 'n_forecasts_c',
	'n_forecasts_d', 'n_forecasts_sage', 'n_forecasts_te', 'ordinal',
	'p_updates', 'stage', 'variance_b', 'variance_c', 'variance_d',
	'variance_human', 'variance_sage', 'variance_te']

	human_feature = pd.read_csv('data/human_features.csv').drop_duplicates(subset=['date', 'ifp_id'], keep='last')
	hf_cols = human_feature.columns.tolist()
	hf_cols.remove('ifp_id')
	hf_cols.remove('date')
	hf_cols.remove('stage')
	hf_cols.remove('p_updates')
	hf_cols.remove('Health/Disease')
	hf_cols.remove('Macroeconomics/Finance')
	hf_cols.remove('Natural Sciences/Climate')
	hf_cols.remove('Other')
	hf_cols.remove('Politics/Intl Relations')
	hf_cols.remove('Technology')
	#if feature_used in human_feature_list:
	#	hf_cols.remove(feature_used)

	human_feature = human_feature.drop(columns=hf_cols)
	print(human_feature.columns)

	ts_feature_list = ['x_acf1', 'x_acf10', 'diff1_acf1', 'diff1_acf10',
		'diff2_acf1', 'diff2_acf10', 'seas_acf1', 'ARCH.LM', 'crossing_points',
		'entropy', 'flat_spots', 'arch_acf', 'garch_acf', 'arch_r2', 'garch_r2',
		'alpha', 'beta', 'hurst', 'lumpiness', 'nonlinearity', 'x_pacf5',
		'diff1x_pacf5', 'diff2x_pacf5', 'seas_pacf', 'nperiods',
		'seasonal_period', 'trend', 'spike', 'linearity', 'curvature', 'e_acf1',
		'e_acf10', 'seasonal_strength', 'peak', 'trough', 'stability',
		'hw_alpha', 'hw_beta', 'hw_gamma', 'unitroot_kpss', 'unitroot_pp',
		'series_length', 'ratio', 'skew']

	ts_feature = pd.read_csv('data/ts_features.csv')
	tf_cols = ts_feature.columns.tolist()
	tf_cols.remove('ifp_id')
	tf_cols.remove('date')
	tf_cols.remove('ratio')
	tf_cols.remove('arch_acf')
	#if feature_used in ts_feature_list:
	#	tf_cols.remove(feature_used)
	ts_feature = ts_feature.drop(columns=tf_cols)
	print(ts_feature.columns)

	n_feature = human_feature.shape[1] + ts_feature.shape[1] - 4
	print('n_feature', n_feature)

	human_dict = {}
	for index, row in human_feature.iterrows():
		ifp_id = row['ifp_id']
		date = row['date']
		if ifp_id not in human_dict:
			human_dict[ifp_id] = {}

		if date in human_dict[ifp_id]:
			pdb.set_trace()
			print('Duplicate feature')
		else:
			human_dict[ifp_id][date] = row.drop(labels=['ifp_id', 'date']).values

	ts_dict = {}
	for index, row in ts_feature.iterrows():
		ifp_id = row['ifp_id']
		date = row['date']
		if ifp_id not in ts_dict:
			ts_dict[ifp_id] = {}

		if date in ts_dict[ifp_id]:
			pdb.set_trace()
			print('Duplicate feature')
		else:
			ts_dict[ifp_id][date] = row.drop(labels=['ifp_id', 'date']).values

	df = pd.read_csv('data/human.csv')
	#df.fillna(0, inplace=True)
	#pdb.set_trace()
	#np.unique(df[df['ifp_id'].isin(ifp_all)]['user_id']).shape
	db = OrderedDict()

	for index, row in df.iterrows():
		date = dateutil.parser.parse(row['date']).replace(tzinfo=None)
		user_id = row['user_id']
		ifp_id = row['ifp_id']

		if ifp_id not in db_answer:
			continue

		num_options = row['num_options']
		option_1 = row['option_1'] / 100.0
		option_2 = row['option_2'] / 100.0
		option_3 = row['option_3'] / 100.0
		option_4 = row['option_4'] / 100.0
		option_5 = row['option_5'] / 100.0

		if num_options == 1:
			num_options = 2

		if ifp_id not in db:
			db[ifp_id] = []

		#cf = get_feature(ifp_id, datetime.strftime(date, "%Y-%m-%d"))
		db[ifp_id].append([date,'human',user_id,ifp_id,num_options,option_1,option_2,option_3,option_4,option_5])# + cf.tolist())

	machine_df = pd.read_csv('data/machine_all.csv').drop_duplicates(subset=['date', 'machine_model', 'ifp_id'], keep='last')
	for index, row in machine_df.iterrows():
		date = dateutil.parser.parse(row['date'])
		machine_model = row['machine_model']
		ifp_id = row['ifp_id']

		if ifp_id not in db_answer:
			continue

		if machine_model not in ('Auto ARIMA', 'M4-Meta', 'Arithmetic RW'):
			continue

		if machine_model not in ('Auto ARIMA', 'M4-Meta', 'Arithmetic RW', 'DS-Holt', 'DS-Holt-damped', 'DS-RW', 'DS-SES', 'ETS', 'Geometric RW', 'M4-Comb', 'Mean', 'NNETAR', 'RW', 'RW-DRIFT', 'RW-SEAS', 'STLM-AR', 'TBATS', 'THETA'):
			continue

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

		if num_options == 1:
			num_options = 2

		if ifp_id not in db:
			pdb.set_trace()
			print("Didn't expect any ifp have human forecast but don't have machine forecast")

		#cf = get_feature(ifp_id, datetime.strftime(date, "%Y-%m-%d"))
		db[ifp_id].append([date,'machine',machine_model,ifp_id,num_options,option_1,option_2,option_3,option_4,option_5])# + cf.tolist())

	for ifp_id in db:
		db[ifp_id].sort(key=lambda x: x[0])


	with open('cache.ckpt', 'wb') as fout:
		pickle.dump([
			db, db_answer, db_dates, human_dict, human_feature, ts_dict, ts_feature
		], fout, pickle.HIGHEST_PROTOCOL)
else:
	with open('cache.ckpt', 'rb') as fin:
		[db, db_answer, db_dates, human_dict, human_feature, ts_dict, ts_feature
		] = pickle.load(fin)
		n_feature = human_feature.shape[1] + ts_feature.shape[1] - 4
		print('n_feature', n_feature)

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

# transforms range
def transform_range(d,ini,fin):
  mi = d[min(d, key=d.get)]
  ma = d[max(d, key=d.get)]
  s=1.
  if (ma - mi)>0:
    s = (fin - ini)/(ma - mi)
  r = {}
  for key, value in d.items():
    r[key] = 1.1 - (value-mi)*s + ini
  return r

def extremize(probs, alpha, ordered):
	c = len(probs)
	cum_ext = []
	if ordered:
		cum_sum = 0.
		for prob in probs:
			p = prob + cum_sum
			if p>1.: #had problems with p=1.00000002
				p=1.0
			cum_ext.append(((c-1.)*max(p, EPS))**alpha/( ((c-1.)*max(p, EPS))**alpha + (c-1.)*(max(1.-p, EPS))**alpha ))
			cum_sum += prob
		last = 0.

		for opt, prob in enumerate(cum_ext):
			cum_ext[opt] = cum_ext[opt] - last
			last = prob
	else:
		for prob in probs:
			#Non-Ordered IFPs:
			cum_ext.append(((c-1.)*max(prob, EPS))**alpha/( ((c-1.)*max(prob, EPS))**alpha + (c-1.)*(max(1.-prob, EPS))**alpha) )

	#normalize
	norm = sum(cum_ext)
	for opt, prob in enumerate(cum_ext):
		cum_ext[opt] = cum_ext[opt]/ norm

	return cum_ext


def M1(data, day, ordered,
		brier_score= defaultdict(lambda: 1.),
		user_activity = defaultdict(lambda: 0.),
		min_activity = 10, # min # of resolved forecasts answered to update accuracy score
		n_most_recent=0.2, # temporal subset
		n_minimum = 10, # the MIN forecasts to keep, this supercedes n_most_recent
		only_human = False, # only use human forecasts
		only_arima = False, #use only arima as machine models
		gamma=2., #accuracy/brier exponent
		alpha=1.3, # the group level extremizing parameter,
		):


	# Step 1:Temporal subsetting
	predinput = [x for x in data if x[0]<=day]
	if only_human:
		predinput = [x for x in predinput if x[1]=='human']
	if only_arima:
		predinput = [x for x in predinput if (x[1]=='human') or (x[2]=='Auto ARIMA' )]
	predinput = sorted([i for i in predinput], key=lambda x: x[0], reverse=True)
	peoplewhoparticipated = set([x[2] for x in predinput])
	forecasts = [] #last forecast of each person
	for person in peoplewhoparticipated:
		s = [x for x in predinput if x[2] == person]
		#get lastest prediction of user
		forecasts.append(s[0])

	#Sort forecasts
	forecasts = sorted([i for i in forecasts], key=lambda x: x[0], reverse=True)
	if len(forecasts)==0:
		return []
	# get their last d-percentile or n_minimum forecasts. Whichever is larger.
	ntokeep = np.ceil(len(forecasts) * n_most_recent)
	ntokeep = max(ntokeep, n_minimum)
	ntokeep = min((len(forecasts)-1),ntokeep)
	ntokeep = int(ntokeep)
	date = forecasts[ntokeep][0]
	# date = datetime.combine(date, time.min)
	# print("Getting forecasts produced on date %s or later ..." % date)
	forecasts = [x for x in forecasts if x[0]>=date]

	# Step 2:Aggregate forecasts
	result =  np.zeros(5)
	for f in forecasts:
		user = f[2]
		num_options = f[4]
		pred = np.array(f[5:])
		if user_activity[user]>=min_activity:
			w = brier_score[user]
		else:
			w = 0.5
		#weighted pred
		# print(brier_score[user])
		# print(w)
		result += (w**gamma)*pred

	# Step 3:Normalize
	result = result[:num_options]
	result /= np.nansum(result)

	#Step 4:Aggregate-level extremization and weighting
	result = extremize(result, alpha, ordered)



	return result


def M0_aggregation(test_input, db_dates,db_answer):

	ifps = test_input.keys()
	results = {}
	for ifp in ifps:
		results[ifp] = []
		for day in db_dates[ifp]:
			pred = M1(test_input[ifp], day,
				ordered= db_answer[ifp][1],
				alpha=1.
				)
			if len(pred)>0:
				score =  brier(pred, db_answer[ifp][0], ordered=db_answer[ifp][1])
				results[ifp].append(score)

	#AVG brier per IFP
	scores = {}
	for ifp in ifps:
		scores[ifp] = np.mean(results[ifp])

	return scores

def M3_aggregation(train_input, test_input, db_dates,db_answer, fold_index):
	brier_score, question2user2brier = get_user_brier(train_input,db_dates,db_answer)
	user_activity = {user: len(ifps) for user, ifps in question2user2brier.items()}
	user_activity = defaultdict(lambda: 0., user_activity)
	brier_score = transform_range(brier_score,0.1,1.)
	brier_score =  defaultdict(lambda: 0.5, brier_score)

	u1 = math.sqrt(6/(n_feature + N_HIDDEN_DIM))
	W1 = torch.FloatTensor(n_feature, N_HIDDEN_DIM).uniform_(-u1, u1)
	W1.requires_grad = True
	b1 = torch.zeros(N_HIDDEN_DIM, requires_grad=True)

	u2 = math.sqrt(6/(N_HIDDEN_DIM + 1))
	W2 = torch.FloatTensor(N_HIDDEN_DIM, 1).uniform_(-u2, u2)
	W2.requires_grad = True
	b2 = torch.zeros(1, requires_grad=True)

	optimizer = torch.optim.Adam([W1, b1, W2, b2], lr=1e-2, weight_decay=1e-5)
	lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=2, verbose=True)

	ifps_train = train_input.keys()
	ifps_test = test_input.keys()

	attempts = []
	for z in range(10):
		results_train = {}
		for ifp in ifps_train:
			results_train[ifp] = []
			for day in db_dates[ifp]:
				feature = get_feature(ifp, datetime.strftime(day, "%Y-%m-%d"))
				alpha = torch.sigmoid(torch.matmul(torch.tanh(torch.matmul(torch.from_numpy(feature.astype(np.float32)), W1) + b1), W2) + b2) * 2
				#print('ifp: {}, day: {}, alpha: {}'.format(ifp, datetime.strftime(day, "%Y-%m-%d"), alpha))
				pred = M1(train_input[ifp], day,
					ordered= db_answer[ifp][1],
					brier_score=brier_score,
					user_activity = user_activity,
					alpha=alpha
				)
				if len(pred)>0:
					score =  brier(pred, db_answer[ifp][0], ordered=db_answer[ifp][1])
					results_train[ifp].append(score)

		#AVG brier per IFP
		scores_train = {}
		for ifp in ifps_train:
			scores_train[ifp] = torch.mean(torch.stack(results_train[ifp]))

		mmdb_train = torch.mean(torch.stack(list(scores_train.values())))

		optimizer.zero_grad()
		mmdb_train.backward()
		optimizer.step()
		#lr_scheduler.step(mmdb_train)

		trend_m3_ary = []
		for i in range(100):
			trend_m3_ary.append([])

		results_test = {}
		for ifp in ifps_test:
			results_test[ifp] = []

			dates = db_dates[ifp]
			start_date = dates[0].replace(hour=0, minute=0, second=0, microsecond=0)
			end_date = dates[-1].replace(hour=23, minute=59, second=59, microsecond=999)
			total_seconds = (end_date-start_date).total_seconds()

			for day in db_dates[ifp]:
				feature = get_feature(ifp, datetime.strftime(day, "%Y-%m-%d"))
				alpha = torch.sigmoid(torch.matmul(torch.tanh(torch.matmul(torch.from_numpy(feature.astype(np.float32)), W1) + b1), W2) + b2) * 2
				#print('ifp: {}, day: {}, alpha: {}'.format(ifp, datetime.strftime(day, "%Y-%m-%d"), alpha))
				pred = M1(test_input[ifp], day,
					ordered= db_answer[ifp][1],
					brier_score=brier_score,
					user_activity = user_activity,
					alpha=alpha
				)
				if len(pred)>0:
					score =  brier(pred, db_answer[ifp][0], ordered=db_answer[ifp][1])
					results_test[ifp].append(score)

					local_seconds = (day.replace(hour=0, minute=0, second=0, microsecond=0)-start_date).total_seconds()
					local_progress = int(np.around(100.0 * local_seconds / total_seconds))
					if local_progress > 99:
						local_progress = 99
					trend_m3_ary[local_progress].append(score.detach().numpy())

		#AVG brier per IFP
		scores_test = {}
		for ifp in ifps_test:
			scores_test[ifp] = torch.mean(torch.stack(results_test[ifp]))

		mmdb_test = torch.mean(torch.stack(list(scores_test.values())))
		print('Epoch: {}, train mmdb: {}, test mmdb: {}'.format(z, mmdb_train, mmdb_test))

		lr_scheduler.step(mmdb_test)
		attempts.append([mmdb_test.detach().numpy(), scores_test, results_test, trend_m3_ary, W1.detach().numpy(), b1.detach().numpy(), W2.detach().numpy(), b2.detach().numpy()])

	best_attempt = sorted(attempts)[0]
	with open('model_m2_feature/{}/model.pickle'.format(fold_index), 'wb') as fout:
		pickle.dump(best_attempt[4:], fout, pickle.HIGHEST_PROTOCOL)

	with open('plot_data/m3_trend_db_{}.pickle'.format(fold_index), 'wb') as fout:
		pickle.dump(best_attempt[3], fout, pickle.HIGHEST_PROTOCOL)

	return best_attempt[1], best_attempt[2]


# Aggregation placeholder
all_ifp = np.asarray(list(db.keys()))

kf = sklearn.model_selection.KFold(shuffle=True, n_splits=5, random_state=2019)
folds = [[all_ifp[f[0]], all_ifp[f[1]]] for f in kf.split(all_ifp)]

#scores_m0 = []
scores_m3 = []
for i in range(5):
	fold_index = i

	ifp_train = folds[fold_index][0]

	ifp_test = folds[fold_index][1]

	#ifp_train = all_ifp
	n_train = len(ifp_train)
	n_test = len(ifp_test)

	train_input = {k: db[k] for k in ifp_train}
	test_input = {k: db[k] for k in ifp_test}

	'''
	#Brier scores for each ifp
	results = M0_aggregation(test_input, db_dates,db_answer)
	with open('plot_data/m0_brier_db_{}.pickle'.format(i), 'wb') as fout:
		pickle.dump(results, fout, pickle.HIGHEST_PROTOCOL)

	#Mean Brier of M0
	print('M0 = ', np.array(list(results.values())).mean())
	scores_m0.append(np.array(list(results.values())).mean())
	'''

	scores, results = M3_aggregation(train_input,test_input, db_dates,db_answer, i)
	scores = {k: v.detach().numpy() for k, v in scores.items()}
	results = {k: [vv.detach().numpy() for vv in v] for k, v in results.items()}

	with open('plot_data/m3_brier_db_{}.pickle'.format(i), 'wb') as fout:
		pickle.dump(scores, fout, pickle.HIGHEST_PROTOCOL)

	#Mean Brier
	print('M3 = ', np.array(list(scores.values())).mean())
	scores_m3.append(np.array(list(scores.values())).mean())

	#Rank
	m3_rank_test = []
	for ifp in ifp_test:
		individual_forecasts = db[ifp]
		answer, is_ordered = db_answer[ifp]
		individual_briers = [brier(p[5:10][:p[4]], answer, is_ordered) for p in individual_forecasts]
		for s in results[ifp]:
			rank = scipy.stats.percentileofscore(individual_briers, s)
			m3_rank_test.append(rank)
	with open('plot_data/m3_rank_db_{}.pickle'.format(i), 'wb') as fout:
		pickle.dump(m3_rank_test, fout, pickle.HIGHEST_PROTOCOL)

pdb.set_trace()
#print(np.mean(scores_m0))
print(np.mean(scores_m3))
#pdb.set_trace()
print('OK')
