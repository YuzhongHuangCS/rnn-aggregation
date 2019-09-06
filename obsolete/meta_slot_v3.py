import dateparser
import pdb
import pickle
import os
import pandas as pd
import sklearn
import sklearn.linear_model
import sklearn.ensemble
import sklearn.svm
import sklearn.neural_network
import numpy as np
from briercompute import brier as brier_compute
import scipy.optimize

slot_1_name = '07_BCD_T_M11b'
slot_2_name = 'Auto ARIMA'

ifp_list_filename = 'ifp_list.pickle'
if os.path.isfile(ifp_list_filename):
	with open(ifp_list_filename, 'rb') as f:
		ifp_train, ifp_test = pickle.load(f)
else:
	ifp_db = {}
	with open('slot_performance.csv') as fin:
		fin.readline()
		for line in fin:
			parts = line.split(',')
			date = dateparser.parse(parts[0])
			ifp = parts[1]
			if ifp in ifp_db:
				prev_date = ifp_db[ifp]
				if date < prev_date:
					ifp_db[ifp] = date
			else:
				ifp_db[ifp] = date

	ifp_list = [(x[1], x[0]) for x in ifp_db.items()]
	ifp_list.sort()

	n_train = int(len(ifp_list) * 0.8)
	n_test = len(ifp_list)-n_train
	ifp_train = set([int(x[1]) for x in ifp_list[:n_train]])
	ifp_test = set([int(x[1]) for x in ifp_list[n_train:]])

	with open(ifp_list_filename, 'wb') as f:
		pickle.dump([ifp_train, ifp_test], f, pickle.HIGHEST_PROTOCOL)


train_df_s1 = []
train_df_s2 = []
test_df_s1 = []
test_df_s2 = []

def is_ordered(opt):
	keywords = ['Less', 'Between', 'More', 'inclusive','less', 'between', 'more']
	if len(opt) == 1: #binary
		return False
	for o in opt:
		if any(x in o for x in keywords):
			return True

	return False

db_answer = {}
df_question = pd.read_csv('dump_questions.csv')
for index, row in df_question.iterrows():
	if row['is_resolved']:
		ifp_id = row['ifp_id']
		resolution = row['resolution']
		options = row.tolist()[-5:]

		clean_options = [x for x in options if type(x) == str]
		try:
			answer = options.index(resolution)
			db_answer[ifp_id] = [answer, is_ordered(clean_options)]
		except ValueError as e:
			pdb.set_trace()
			print(e)

with open('slot_performance.csv') as fin:
	fin.readline()
	for line in fin:
		parts = line.strip().split(',')
		date = parts[0].split(' ')[0]
		ifp = int(parts[1])
		slot = parts[2]
		brier = float(parts[3])
		prob = [float(x) for x in parts[-5:]]
		if ifp in ifp_train:
			if slot == slot_1_name:
				train_df_s1.append([ifp, date, brier, max(prob), db_answer[ifp][0], db_answer[ifp][0], prob[0], prob[1], prob[2], prob[3], prob[4]])

			if slot == slot_2_name:
				train_df_s2.append([ifp, date, brier, prob[0], prob[1], prob[2], prob[3], prob[4]])

		if ifp in ifp_test:
			if slot == slot_1_name:
				test_df_s1.append([ifp, date, brier, max(prob), db_answer[ifp][0], db_answer[ifp][0], prob[0], prob[1], prob[2], prob[3], prob[4]])

			if slot == slot_2_name:
				test_df_s2.append([ifp, date, brier, prob[0], prob[1], prob[2], prob[3], prob[4]])

train_df_s1 = pd.DataFrame(train_df_s1).rename(columns={0: 'ifp_id', 1: 'date', 2: 'slot1', 3: 'max_prob', 4: 'answer', 5: 'weight'})
train_df_s2 = pd.DataFrame(train_df_s2).rename(columns={0: 'ifp_id', 1: 'date', 2: 'slot2'})
train_df = pd.merge(train_df_s1, train_df_s2, how='inner', on=['ifp_id', 'date'])
test_df_s1 = pd.DataFrame(test_df_s1).rename(columns={0: 'ifp_id', 1: 'date', 2: 'slot1', 3: 'max_prob', 4: 'answer', 5: 'weight'})
test_df_s2 = pd.DataFrame(test_df_s2).rename(columns={0: 'ifp_id', 1: 'date', 2: 'slot2'})
test_df = pd.merge(test_df_s1, test_df_s2, how='inner', on=['ifp_id', 'date'])

human_feature = pd.read_csv('human_features.csv')
ts_feature = pd.read_csv('ts_features_rctb.csv')

def get_brier(w, human_prob, machine_prob, answer_index, ordered):
	return brier_compute(w*human_prob + (1-w)*machine_prob, answer_index, ordered)

for index, row in train_df.iterrows():
	answer_index = row['answer']
	human_prob = row.values[6:11].astype(float)
	machine_prob = row.values[12:17].astype(float)
	ordered = db_answer[row['ifp_id']]
	opt = scipy.optimize.minimize(get_brier, args=(human_prob, machine_prob, answer_index, ordered), x0=(0.5,), bounds=((0, 1),))
	#pdb.set_trace()
	train_df.loc[index, 'weight'] = opt.x

for index, row in test_df.iterrows():
	answer_index = row['answer']
	human_prob = row.values[6:11].astype(float)
	machine_prob = row.values[12:17].astype(float)
	ordered = db_answer[row['ifp_id']]
	opt = scipy.optimize.minimize(get_brier, args=(human_prob, machine_prob, answer_index, ordered), x0=(1,), bounds=((-1, 2),))
	test_df.loc[index, 'weight'] = opt.x

train_df = pd.merge(train_df, human_feature, how='inner', on=['ifp_id', 'date'])
train_df = pd.merge(train_df, ts_feature, how='inner', on=['ifp_id', 'date'])

test_df = pd.merge(test_df, human_feature, how='inner', on=['ifp_id', 'date'])
test_df = pd.merge(test_df, ts_feature, how='inner', on=['ifp_id', 'date'])

train_X = train_df.drop(columns=['ifp_id', 'date', 'slot1', 'slot2', 'answer', 'weight']).values
train_Y = train_df['weight'].values

reg = sklearn.ensemble.GradientBoostingRegressor(loss='ls', n_estimators=1000)
reg.fit(train_X, train_Y)

pdb.set_trace()
test_X = test_df.drop(columns=['ifp_id', 'date', 'slot1', 'slot2', 'answer', 'weight']).values
test_Y = test_df['weight'].values
test_Y_weight = reg.predict(test_X)
test_Y_brier = []

for i, w in enumerate(test_Y_weight):
	if w > 0.4:
		w = 1
	row = test_df.iloc[i]
	ifp_id = row['ifp_id']
	human_prob = row.values[6:11].astype(float)
	machine_prob = row.values[12:17].astype(float)
	comb = np.clip(w*human_prob + (1-w)*machine_prob, 0, 1)
	comb /= sum(comb)

	answer_2, ordered = db_answer[ifp_id]
	answer = row['answer']
	assert answer == answer_2

	new_brier = brier_compute(comb, answer, ordered)
	test_Y_brier.append(new_brier)

test_Y_brier = np.asarray(test_Y_brier)
db_brier = {}

for ifp in test_df['ifp_id']:
	index = test_df.index[test_df['ifp_id'] == ifp]
	slot1 = test_df.iloc[index]['slot1'].values
	slot2 = test_df.iloc[index]['slot2'].values
	test_ideal = np.amin([slot1, slot2], axis=0)

	test_predict = test_Y_brier[index]
	db_brier[ifp] = [np.mean(slot1), np.mean(slot2), np.mean(test_ideal), np.mean(test_predict)]
	print([np.mean(slot1), np.mean(slot2), np.mean(test_ideal), np.mean(test_predict)])

with open('briers.csv', 'w') as fout:
	fout.write('slot1,slot2,ideal,predict\n')
	for row in db_brier.values():
		fout.write(','.join([str(x) for x in row]) + '\n')


briers = np.asarray(list(db_brier.values()))
print('slot1', 'slot2', 'ideal', 'predict')
print(np.mean(briers, axis=0))
#pdb.set_trace()
print(reg.score(train_X, train_Y))
print(reg.score(test_X, test_Y))
print('OK')
