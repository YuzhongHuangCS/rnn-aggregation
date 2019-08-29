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
import random
import copy
from briercompute import brier, get_user_brier
from datetime import datetime, timedelta
import math


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
			is_ordered = row['is_ordinal']

			clean_options = [x for x in options if type(x) == str]
			try:
				answer = options.index(resolution)
				if ifp_id in db_answer:
					pdb.set_trace()
					print(ifp_id)
				else:
					db_answer[ifp_id] = [answer, is_ordered]

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


df = pd.read_csv('data/human.csv')
df = df.loc[df.ifp_id.isin(db_dates.keys())]

# Descriptive statistics of data
duration = []
for ifp, days in db_dates.items():
	delta = (max(days) - min(days)).days
	print(ifp, delta)
	duration.append(delta)

print('forecasts per question')
df_grouped = df.groupby(['ifp_id'])['user_id'].count()
print(df_grouped.describe())

print('forecasters per question')
df_grouped = df.groupby(['ifp_id'])['user_id'].nunique()
print(df_grouped.describe())

print('forecasts per user')
df_grouped = df.groupby(['user_id'])['ifp_id'].count()
print(df_grouped.describe())

print('# forecasters', len(df.user_id.unique()) )
print('# forecasts', len(df.user_id) )


print('# if IFPS = ', len(duration))
print('min duration = ', np.min(duration))
print('median duration = ', np.median(duration))
print('mean duration = ', np.mean(duration))
print('max duration = ', np.max(duration))
print('Std duration = ', np.std(duration))


#sample prediction ifp


