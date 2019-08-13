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
# import tensorflow as tf
import random
import copy
from briercompute import brier
from datetime import datetime, timedelta
import math

N_RNN_DIM = 32
N_EMB_DIM = 8

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


	human_feature = pd.read_csv('data/human_features.csv').drop_duplicates(subset=['date', 'ifp_id'], keep='last')
	ts_feature = pd.read_csv('data/ts_features.csv')

	#n_feature = human_feature.shape[1] + ts_feature.shape[1] - 4
	n_feature = 0

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
			db, db_answer, db_dates,
		], fout, pickle.HIGHEST_PROTOCOL)
else:
	with open('cache.ckpt', 'rb') as fin:
		[db, db_answer, db_dates,
		] = pickle.load(fin)

# db[ifp_id].append([date,'human',user_id,ifp_id,num_options,option_1,option_2,option_3,option_4,option_5])
def M0(data, day,
		n_most_recent=0.3, # temporal subset
		n_minimum = 10, # the MIN forecasts to keep, this supercedes n_most_recent 
		only_human = False,
		):

	
	# Step 1:Temporal subsetting
	predinput = [x for x in data if x[0]<=day]
	if only_human:
		predinput = [x for x in predinput if x[1]=='human']
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
		w = 1.0 #uniform weights
		result += w*pred

	# Step 3:Normalize
	result = result[:num_options]
	result /= np.nansum(result)



	return result


def M0_aggregation(test_input, db_dates,db_answer):

	ifps = test_input.keys()
	results = {}
	for ifp in ifps:
		results[ifp] = []
		for day in db_dates[ifp]:
			pred = M0(test_input[ifp], day)
			if len(pred)>0:
				score =  brier(pred, db_answer[ifp][0], ordered=db_answer[ifp][1])
				results[ifp].append(score)

	#AVG brier per IFP
	scores = {}
	for ifp in ifps:
		scores[ifp] = np.mean(results[ifp])

	return scores





# Aggregation placeholder
all_ifp = np.asarray(list(db.keys()))

kf = sklearn.model_selection.KFold(shuffle=True, n_splits=5, random_state=2019)
folds = [[all_ifp[f[0]], all_ifp[f[1]]] for f in kf.split(all_ifp)]
fold_index = 0

ifp_train = folds[fold_index][0]
ifp_test = folds[fold_index][1]

#ifp_train = all_ifp
n_train = len(ifp_train)
n_test = len(ifp_test)

train_input = {k: db[k] for k in ifp_train}
test_input = {k: db[k] for k in ifp_test}


#Brier scores for each ifp
results = M0_aggregation(test_input, db_dates,db_answer)
#Mean Brier
print(np.array(list(results.values())).mean())


print('OK')
