import os

import pandas as pd
import pdb
import sys
import dateutil.parser
import numpy as np
from collections import OrderedDict, Counter, defaultdict
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
from utils import is_ordered

N_HIDDEN_DIM = 16
EPS = 1e-8

if len(sys.argv) >= 2:
	feature_used = sys.argv[1:]
else:
	feature_used = ['None', ]

def is_ordered(opt):
	keywords = ['Less', 'Between', 'More', 'inclusive','less', 'between', 'more']
	if len(opt) <= 2: #binary
		return False
	for o in opt:
		if any(x in o for x in keywords):
			return True

	return False

if True or not os.path.exists('cache.ckpt'):
	print('Reading data')
	db_answer = {}
	db_boundary = {}

	for filename in ('data/dump_questions_rcta.csv', 'data/dump_questions_rctb.csv', 'data/dump_questions_rctc.csv'):
		df_question = pd.read_csv(filename)
		for index, row in df_question.iterrows():
			if row['is_resolved'] and (not row['is_voided']):
				if filename == 'data/dump_questions_rctc.csv':
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

				except ValueError as e:
					pdb.set_trace()
					print(e)

	human_feature_list = ['n_forecasts_te', 'variance_sage', 'n_forecasts_d', 'n_forecasts_sage', 'entropy_b', 'entropy_d', 'entropy_te', 'n_forecasts_b', 'entropy_c', 'Technology', 'variance_b', 'variance_d', 'Other', 'n_forecasts_c', 'stage', 'entropy_sage', 'n_forecasts', 'Politics/Intl Relations', 'Macroeconomics/Finance', 'variance_te', 'variance_c', 'variance_human', 'entropy_human', 'ordinal', 'Natural Sciences/Climate']
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
	for t_f in (ts_feature, ts_feature_rctc):
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

	db = defaultdict(list)
	df_ary = []
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

			df_ary.append([ifp_id, user_id])
			db[ifp_id].append([date,'human',user_id,ifp_id,num_options,option_1,option_2,option_3,option_4,option_5])

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

		db[ifp_id].append([date,'machine',machine_model,ifp_id,num_options,option_1,option_2,option_3,option_4,option_5])

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

	df_all = pd.DataFrame(df_ary, columns=['ifp_id', 'user_id'])
	df_all = df_all.loc[df_all.ifp_id.isin(db_dates.keys())]
	# Descriptive statistics of data
	duration = []
	for ifp, days in db_dates.items():
		delta = (max(days) - min(days)).days
		if delta == 0:
			delta = 1
		print(ifp, delta)
		duration.append(delta)

	print('forecasts per question')
	df_grouped = df_all.groupby(['ifp_id'])['user_id'].count()
	print(df_grouped.describe())

	print('forecasters per question')
	df_grouped = df_all.groupby(['ifp_id'])['user_id'].nunique()
	print(df_grouped.describe())

	print('forecasts per user')
	df_grouped = df_all.groupby(['user_id'])['ifp_id'].count()
	print(df_grouped.describe())

	print('# forecasters', len(df_all.user_id.unique()) )
	print('# forecasts', len(df_all.user_id) )


	print('# if IFPS = ', len(duration))
	print('min duration = ', np.min(duration))
	print('median duration = ', np.median(duration))
	print('mean duration = ', np.mean(duration))
	print('max duration = ', np.max(duration))
	print('Std duration = ', np.std(duration))

	with open('cache.ckpt', 'wb') as fout:
		pickle.dump([
			db, db_answer, db_dates,
		], fout, pickle.HIGHEST_PROTOCOL)
else:
	with open('cache.ckpt', 'rb') as fin:
		[db, db_answer, db_dates,
		] = pickle.load(fin)


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


def M0_aggregation(test_input, db_dates,db_answer,fold_index):
	trend_m0_ary = []
	for i in range(100):
		trend_m0_ary.append([])

	ifps = test_input.keys()
	results = {}
	predictions = {'score':[], 'true':[]}
	for ifp in ifps:
		results[ifp] = []

		dates = db_dates[ifp]
		start_date = dates[0].replace(hour=0, minute=0, second=0, microsecond=0)
		end_date = dates[-1].replace(hour=23, minute=59, second=59, microsecond=999)
		total_seconds = (end_date-start_date).total_seconds()

		for day in db_dates[ifp]:
			pred = M1(test_input[ifp], day,
				ordered= db_answer[ifp][1],
				alpha=1.
				)
			if len(pred)>0:
				score =  brier(pred, db_answer[ifp][0], ordered=db_answer[ifp][1])
				results[ifp].append(score)

				predictions['score'] += pred
				true_pred = list(np.zeros(len(pred)))
				true_pred[db_answer[ifp][0]] = 1
				predictions['true']+= true_pred

				local_seconds = (day.replace(hour=0, minute=0, second=0, microsecond=0)-start_date).total_seconds()
				local_progress = int(np.around(100.0 * local_seconds / total_seconds))
				if local_progress > 99:
					local_progress = 99
				trend_m0_ary[local_progress].append(score)
			else:
				pdb.set_trace()
				print('Unexpected. Forecast dates have been adjusted for activity dates')

	with open('plot_data/m0_trend_db_{}.pickle'.format(fold_index), 'wb') as fout:
		pickle.dump(trend_m0_ary, fout, pickle.HIGHEST_PROTOCOL)

	#AVG brier per IFP
	scores = {}
	for ifp in ifps:
		scores[ifp] = np.mean(results[ifp])

	return scores, results, predictions

def M1_aggregation(test_input, db_dates,db_answer,fold_index):
	trend_m1_ary = []
	for i in range(100):
		trend_m1_ary.append([])

	ifps = test_input.keys()
	results = {}
	predictions = {'score':[], 'true':[]}
	for ifp in ifps:
		results[ifp] = []

		dates = db_dates[ifp]
		start_date = dates[0].replace(hour=0, minute=0, second=0, microsecond=0)
		end_date = dates[-1].replace(hour=23, minute=59, second=59, microsecond=999)
		total_seconds = (end_date-start_date).total_seconds()

		for day in db_dates[ifp]:
			pred = M1(test_input[ifp], day,
				ordered= db_answer[ifp][1],
				)
			if len(pred)>0:
				score =  brier(pred, db_answer[ifp][0], ordered=db_answer[ifp][1])
				results[ifp].append(score)

				predictions['score'] += pred
				true_pred = list(np.zeros(len(pred)))
				true_pred[db_answer[ifp][0]] = 1
				predictions['true']+= true_pred

				local_seconds = (day.replace(hour=0, minute=0, second=0, microsecond=0)-start_date).total_seconds()
				local_progress = int(np.around(100.0 * local_seconds / total_seconds))
				if local_progress > 99:
					local_progress = 99
				trend_m1_ary[local_progress].append(score)
			else:
				pdb.set_trace()
				print('Unexpected. Forecast dates have been adjusted for activity dates')

	with open('plot_data/m1_trend_db_{}.pickle'.format(fold_index), 'wb') as fout:
		pickle.dump(trend_m1_ary, fout, pickle.HIGHEST_PROTOCOL)

	#AVG brier per IFP
	scores = {}
	for ifp in ifps:
		scores[ifp] = np.mean(results[ifp])

	return scores, results, predictions

def M2_aggregation(train_input,test_input, db_dates,db_answer,fold_index):
	trend_m2_ary = []
	for i in range(100):
		trend_m2_ary.append([])

	brier_score, question2user2brier = get_user_brier(train_input,db_dates,db_answer)
	user_activity = {user: len(ifps) for user, ifps in question2user2brier.items()}
	user_activity = defaultdict(lambda: 0., user_activity)
	brier_score = transform_range(brier_score,0.1,1.)
	brier_score =  defaultdict(lambda: 0.5, brier_score)

	ifps = test_input.keys()
	results = {}
	predictions = {'score':[], 'true':[]}
	for ifp in ifps:
		results[ifp] = []

		dates = db_dates[ifp]
		start_date = dates[0].replace(hour=0, minute=0, second=0, microsecond=0)
		end_date = dates[-1].replace(hour=23, minute=59, second=59, microsecond=999)
		total_seconds = (end_date-start_date).total_seconds()

		for day in db_dates[ifp]:
			#pdb.set_trace()
			pred = M1(test_input[ifp], day,
				ordered= db_answer[ifp][1],
				brier_score=brier_score,
				user_activity = user_activity
				)
			if len(pred)>0:
				score =  brier(pred, db_answer[ifp][0], ordered=db_answer[ifp][1])
				results[ifp].append(score)

				predictions['score'] += pred
				true_pred = list(np.zeros(len(pred)))
				true_pred[db_answer[ifp][0]] = 1
				predictions['true']+= true_pred

				local_seconds = (day.replace(hour=0, minute=0, second=0, microsecond=0)-start_date).total_seconds()
				local_progress = int(np.around(100.0 * local_seconds / total_seconds))
				if local_progress > 99:
					local_progress = 99
				trend_m2_ary[local_progress].append(score)
			else:
				pdb.set_trace()
				print('Unexpected. Forecast dates have been adjusted for activity dates')

	with open('plot_data/m2_trend_db_{}.pickle'.format(fold_index), 'wb') as fout:
		pickle.dump(trend_m2_ary, fout, pickle.HIGHEST_PROTOCOL)

	#AVG brier per IFP
	scores = {}
	for ifp in ifps:
		scores[ifp] = np.mean(results[ifp])

	return scores, results, predictions

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
				else:
					set_trace()
					print('Unexpected. Forecast dates have been adjusted for activity dates')

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
		predictions_test = {'score':[], 'true':[]}
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

					pred_numpy = np.asarray([x.detach().numpy() for x in pred]).squeeze()
					predictions_test['score'] += pred_numpy.tolist()
					true_pred = list(np.zeros(len(pred_numpy)))
					true_pred[db_answer[ifp][0]] = 1
					predictions_test['true'] += true_pred

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
		attempts.append([mmdb_test.detach().numpy(), scores_test, results_test, predictions_test, trend_m3_ary, W1.detach().numpy(), b1.detach().numpy(), W2.detach().numpy(), b2.detach().numpy()])

	best_attempt = sorted(attempts)[0]
	with open('model_m3/{}/model.pickle'.format(fold_index), 'wb') as fout:
		pickle.dump(best_attempt[5:], fout, pickle.HIGHEST_PROTOCOL)

	with open('plot_data/m3_trend_db_{}.pickle'.format(fold_index), 'wb') as fout:
		pickle.dump(best_attempt[4], fout, pickle.HIGHEST_PROTOCOL)

	return best_attempt[1], best_attempt[2], best_attempt[3]

# Aggregation placeholder
all_ifp = np.asarray(list(db.keys()))

kf = sklearn.model_selection.KFold(shuffle=True, n_splits=5, random_state=1)
folds = [[all_ifp[f[0]], all_ifp[f[1]]] for f in kf.split(all_ifp)]

scores_m0 = []
scores_m1 = []
scores_m2 = []
scores_m3 = []
for i in range(5):
	fold_index = i

	ifp_train = folds[fold_index][0]

	ifp_test = folds[fold_index][1]

	#ifp_train = all_ifp
	n_train = len(ifp_train)
	n_test = len(ifp_test)
	print('n_train', n_train, 'n_test', n_test)

	train_input = {k: db[k] for k in ifp_train}
	test_input = {k: db[k] for k in ifp_test}

	#M0
	scores, results, predictions = M0_aggregation(test_input, db_dates,db_answer,i)
	with open('plot_data/m0_brier_db_{}.pickle'.format(i), 'wb') as fout:
		pickle.dump(scores, fout, pickle.HIGHEST_PROTOCOL)
	with open('plot_data/m0_predictions_db_{}.pickle'.format(i), 'wb') as fout:
		pickle.dump(predictions, fout, pickle.HIGHEST_PROTOCOL)
	#Mean Brier
	print('M0 = ', np.array(list(scores.values())).mean())
	scores_m0.append(np.array(list(scores.values())).mean())

	#Rank
	m0_rank_test = []
	for ifp in ifp_test:
		individual_forecasts = db[ifp]
		answer, is_ordered = db_answer[ifp]
		individual_briers = [brier(p[5:10][:p[4]], answer, is_ordered) for p in individual_forecasts]
		for s in results[ifp]:
			rank = scipy.stats.percentileofscore(individual_briers, s)
			m0_rank_test.append(rank)
	with open('plot_data/m0_rank_db_{}.pickle'.format(i), 'wb') as fout:
		pickle.dump(m0_rank_test, fout, pickle.HIGHEST_PROTOCOL)

	#M1
	scores, results, predictions = M1_aggregation(test_input, db_dates,db_answer,i)
	with open('plot_data/m1_brier_db_{}.pickle'.format(i), 'wb') as fout:
		pickle.dump(scores, fout, pickle.HIGHEST_PROTOCOL)
	with open('plot_data/m1_predictions_db_{}.pickle'.format(i), 'wb') as fout:
		pickle.dump(predictions, fout, pickle.HIGHEST_PROTOCOL)

	#Mean Brier
	print('M1 = ', np.array(list(scores.values())).mean())
	scores_m1.append(np.array(list(scores.values())).mean())

	#Rank
	m1_rank_test = []
	for ifp in ifp_test:
		individual_forecasts = db[ifp]
		answer, is_ordered = db_answer[ifp]
		individual_briers = [brier(p[5:10][:p[4]], answer, is_ordered) for p in individual_forecasts]
		for s in results[ifp]:
			rank = scipy.stats.percentileofscore(individual_briers, s)
			m1_rank_test.append(rank)
	with open('plot_data/m1_rank_db_{}.pickle'.format(i), 'wb') as fout:
		pickle.dump(m1_rank_test, fout, pickle.HIGHEST_PROTOCOL)

	#M2
	scores, results, predictions = M2_aggregation(train_input, test_input, db_dates,db_answer,i)
	with open('plot_data/m2_brier_db_{}.pickle'.format(i), 'wb') as fout:
		pickle.dump(scores, fout, pickle.HIGHEST_PROTOCOL)
	with open('plot_data/m2_predictions_db_{}.pickle'.format(i), 'wb') as fout:
		pickle.dump(predictions, fout, pickle.HIGHEST_PROTOCOL)

	#Mean Brier
	print('M2 = ', np.array(list(scores.values())).mean())
	scores_m2.append(np.array(list(scores.values())).mean())

	#Rank
	m2_rank_test = []
	for ifp in ifp_test:
		individual_forecasts = db[ifp]
		answer, is_ordered = db_answer[ifp]
		individual_briers = [brier(p[5:10][:p[4]], answer, is_ordered) for p in individual_forecasts]
		for s in results[ifp]:
			rank = scipy.stats.percentileofscore(individual_briers, s)
			m2_rank_test.append(rank)
	with open('plot_data/m2_rank_db_{}.pickle'.format(i), 'wb') as fout:
		pickle.dump(m2_rank_test, fout, pickle.HIGHEST_PROTOCOL)

	#M3
	scores, results, predictions = M3_aggregation(train_input, test_input, db_dates,db_answer, i)
	scores = {k: v.detach().numpy() for k, v in scores.items()}
	results = {k: [vv.detach().numpy() for vv in v] for k, v in results.items()}

	with open('plot_data/m3_brier_db_{}.pickle'.format(i), 'wb') as fout:
		pickle.dump(scores, fout, pickle.HIGHEST_PROTOCOL)
	with open('plot_data/m3_predictions_db_{}.pickle'.format(i), 'wb') as fout:
		pickle.dump(predictions, fout, pickle.HIGHEST_PROTOCOL)
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
print(np.mean(scores_m0))
print(np.mean(scores_m1))
print(np.mean(scores_m2))
print(np.mean(scores_m3))
#pdb.set_trace()
print('OK')
