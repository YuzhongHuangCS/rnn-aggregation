import os

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
from utils import is_ordered
import re
import json
import requests

templates = []
for line in open('templates.txt', encoding='utf-8'):
	data = json.loads(line)
	template = data['ifp_template']

	exp = template.replace('?', r'\?').replace('+', r'\+').replace('$', r'\$').replace('(', r'\(').replace(')', r'\)')
	exp2 = re.sub(r'__([^ ])+ ', r'(.+)', exp)
	exp3 = re.sub(r'\[([^]]+)\]+', r'(.+)', exp2)

	#print(exp)
	#print(exp2)
	print(exp3)
	regex = re.compile(exp3, re.IGNORECASE)
	templates.append([regex, data.get('last_event_date_key', data['ylabel'])])

db_answer = {}
db_dates = {}
db_source = {}

print('Reading questions')
ifp_blacklist = (2988, )
for filename in ('data/dump_questions_rctb.csv', 'data/dump_questions_rctc.csv'):
	df_question = pd.read_csv(filename)
	for index, row in df_question.iterrows():
		if row['is_resolved'] and (not row['is_voided']):
			if filename == 'data/dump_questions_rctc.csv':
				ifp_id = row['hfc_id']
			else:
				ifp_id = row['ifp_id']

			if ifp_id in ifp_blacklist:
				continue

			resolution = row['resolution']
			options = row.tolist()[-5:]

			clean_options = [x for x in options if type(x) == str]
			try:
				answer = options.index(resolution)
				if ifp_id in db_answer:
					pdb.set_trace()
					print('Duplicated ifp')
				else:
					matches = []
					for regex, key in templates:
						m = regex.match(row['title'])
						if m:
							matches.append([regex, key])

					if len(set(x[1] for x in matches)) == 1:
						db_source[ifp_id] = matches[0][1]
					else:
						if len(matches) == 0:
							# check whether this ifp have machine forecasts
							res = requests.get(f'http://dig:dIgDiG@ec2-52-53-214-202.us-west-1.compute.amazonaws.com/es/ifp_predictions/predictions/{ifp_id}').json()
							if res['found'] == False:
								continue
						if 'What will be the daily closing price of' in row['title']:
							# use the more specific template
							matches = sorted(matches, key=lambda x: len(x[0].pattern), reverse=True)
							db_source[ifp_id] = matches[0][1]
						else:
							pdb.set_trace()
							print('Unexpected template matching')

					if filename == 'data/dump_questions_rcta.csv':
						start_date = dateutil.parser.parse(row['start_date']).replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)
						end_date = dateutil.parser.parse(row['end_date']).replace(hour=23, minute=59, second=59, microsecond=999, tzinfo=None)
					else:
						start_date = dateutil.parser.parse(row['scoring_start_time']).replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)
						end_date = dateutil.parser.parse(row['scoring_end_time']).replace(hour=23, minute=59, second=59, microsecond=999, tzinfo=None)

					resolved_date = dateutil.parser.parse(row['resolved_date']).replace(hour=23, minute=59, second=59, microsecond=999, tzinfo=None)
					end_date = min(end_date, resolved_date)

					forecast_dates = []
					forecast_date = start_date
					while forecast_date <= end_date:
						forecast_dates.append(forecast_date.replace(hour=23, minute=59, second=59, microsecond=999))
						forecast_date += timedelta(days=1)
					db_dates[ifp_id] = forecast_dates
					db_answer[ifp_id] = [answer, is_ordered(clean_options)]
			except ValueError as e:
				pdb.set_trace()
				print(e)

print('Reading human forecasts')
# db -> ifp -> date -> [human forecasts, machine forecasts]
db = defaultdict(lambda: defaultdict(lambda: [[], [], [], []]))
for filename in ('data/dump_user_forecasts_rctb.csv', 'data/dump_user_forecasts_rctc.csv'):
	df = pd.read_csv(filename)
	for index, row in df.iterrows():
		date = dateutil.parser.parse(row['date']).replace(tzinfo=None)

		if filename == 'data/dump_user_forecasts_rctc.csv':
			ifp_id = row['hfc_id']
		else:
			ifp_id = row['ifp_id']

		if ifp_id not in db_answer:
			continue

		num_options = row['num_options']
		prob = row[['option_1', 'option_2', 'option_3', 'option_4', 'option_5']].values / 100.0

		if num_options == 1:
			num_options = 2

		answer, is_ordered = db_answer[ifp_id]
		score = brier(prob[:num_options], answer, is_ordered)
		correct = (np.argmax(prob[:num_options]) == answer)

		date_str = date.strftime('%Y-%m-%d')
		db[ifp_id][date_str][0].append(score)
		db[ifp_id][date_str][2].append(correct)

print('Reading machine forecasts')
machine_df = pd.read_csv('data/machine_all.csv').drop_duplicates(subset=['date', 'machine_model', 'ifp_id'], keep='last')
for index, row in machine_df.iterrows():
	date = dateutil.parser.parse(row['date']).replace(tzinfo=None)
	machine_model = row['machine_model']
	ifp_id = row['ifp_id']

	if ifp_id not in db_answer:
		continue

	if machine_model != 'Auto ARIMA':
		continue

	num_options = row['num_options']
	if num_options == 0:
		continue

	if num_options == 1:
		num_options = 2

	prob = row[['option_1', 'option_2', 'option_3', 'option_4', 'option_5']].values

	if ifp_id not in db:
		pdb.set_trace()
		print("Didn't expect any ifp have human forecast but don't have machine forecast")

	answer, is_ordered = db_answer[ifp_id]
	score = brier(prob[:num_options], answer, is_ordered)
	correct = (np.argmax(prob[:num_options]) == answer)

	date_str = date.strftime('%Y-%m-%d')
	db[ifp_id][date_str][1].append(score)
	db[ifp_id][date_str][3].append(correct)

# db_ratio -> ifp -> date -> [#better, #worse]
# If there is no human or machine forecasts on a day, it will be skipeed
print('Calculating ratio')
db_ratio = defaultdict(dict)
for ifp_id in db:
	for forecast_date in db_dates[ifp_id]:
		date_str = forecast_date.strftime('%Y-%m-%d')
		human_scores, machine_scores, human_corrects, machine_corrects = db[ifp_id][date_str]
		if len(human_scores) == 0 or len(machine_scores) == 0:
			continue

		if len(machine_scores) != 1:
			pdb.set_trace()
			print("Didn't expect two machine forecasts on a day")

		n_better = sum(np.asarray(human_scores) <= machine_scores[0])
		n_worse = len(human_scores) - n_better

		db_ratio[ifp_id][date_str] = [n_better, n_worse, sum(human_corrects), sum(machine_corrects)]

with open('indicator.csv', 'w') as fout:
	fout.write('source,IFP,date,n_better,n_worse,n_human_correct,n_machine_correct\n')
	for ifp_id in db_ratio:
		for date_str in db_ratio[ifp_id]:
			n_better, n_worse, n_human_correct, n_machine_correct = db_ratio[ifp_id][date_str]
			fout.write(f'{db_source[ifp_id]},{ifp_id},{date_str},{n_better},{n_worse},{n_human_correct},{n_machine_correct}\n')

pdb.set_trace()
print('All done')
