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
import matplotlib.pyplot as plt

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
				train_df_s1.append([ifp, date, brier, db_answer[ifp][0], db_answer[ifp][1], prob[0], prob[1], prob[2], prob[3], prob[4]])

			if slot == slot_2_name:
				train_df_s2.append([ifp, date, brier, prob[0], prob[1], prob[2], prob[3], prob[4]])

		if ifp in ifp_test:
			if slot == slot_1_name:
				test_df_s1.append([ifp, date, brier, db_answer[ifp][0], db_answer[ifp][1], prob[0], prob[1], prob[2], prob[3], prob[4]])

			if slot == slot_2_name:
				test_df_s2.append([ifp, date, brier, prob[0], prob[1], prob[2], prob[3], prob[4]])

train_df_s1 = pd.DataFrame(train_df_s1).rename(columns={0: 'ifp_id', 1: 'date', 2: 'slot1', 3: 'answer', 4: 'ordered'})
train_df_s2 = pd.DataFrame(train_df_s2).rename(columns={0: 'ifp_id', 1: 'date', 2: 'slot2'})
train_df = pd.merge(train_df_s1, train_df_s2, how='inner', on=['ifp_id', 'date'])
test_df_s1 = pd.DataFrame(test_df_s1).rename(columns={0: 'ifp_id', 1: 'date', 2: 'slot1', 3: 'answer', 4: 'ordered'})
test_df_s2 = pd.DataFrame(test_df_s2).rename(columns={0: 'ifp_id', 1: 'date', 2: 'slot2'})
test_df = pd.merge(test_df_s1, test_df_s2, how='inner', on=['ifp_id', 'date'])

human_feature = pd.read_csv('human_features.csv')
ts_feature = pd.read_csv('ts_features_rctb.csv')

def get_brier(w, human_prob, machine_prob, answer_index, ordered):
	return brier_compute(w*human_prob + (1-w)*machine_prob, answer_index, ordered)

for index, row in train_df.iterrows():
	answer_index = row['answer']
	human_prob = row.values[5:10].astype(float)
	machine_prob = row.values[11:16].astype(float)
	ordered = db_answer[row['ifp_id']]

for index, row in test_df.iterrows():
	answer_index = row['answer']
	human_prob = row.values[5:10].astype(float)
	machine_prob = row.values[11:16].astype(float)
	ordered = db_answer[row['ifp_id']]

#train_df = pd.merge(train_df, human_feature, how='inner', on=['ifp_id', 'date'])
#train_df = pd.merge(train_df, ts_feature, how='inner', on=['ifp_id', 'date'])

#test_df = pd.merge(test_df, human_feature, how='inner', on=['ifp_id', 'date'])
#test_df = pd.merge(test_df, ts_feature, how='inner', on=['ifp_id', 'date'])

train_X = train_df.drop(columns=['ifp_id', 'date', 'slot1', 'slot2', 'answer', 'ordered']).values
train_Y = train_df['answer'].values


#clf = sklearn.neural_network.MLPClassifier(alpha=0.01, hidden_layer_sizes=(1000, 50))
#clf = sklearn.linear_model.LogisticRegression(penalty='l1', solver='saga', multi_class='multinomial', C=0.5)
clf = sklearn.ensemble.GradientBoostingClassifier(n_estimators=100)
clf.fit(train_X, train_Y)

#pdb.set_trace()
test_X = test_df.drop(columns=['ifp_id', 'date', 'slot1', 'slot2', 'answer', 'ordered']).values
test_Y = test_df['answer'].values

with open('dataset.pickle', 'wb') as f:
	pickle.dump([train_df, test_df], f, pickle.HIGHEST_PROTOCOL)


test_Y_prob = clf.predict_proba(test_X)
test_Y_brier = []

for i, v in enumerate(test_Y_prob):
	ifp_id = test_df.iloc[i]['ifp_id']
	answer_1 = test_Y[i]
	answer_2, ordered = db_answer[ifp_id]
	assert answer_1 == answer_2

	new_brier = brier_compute(v, answer_1, ordered)
	test_Y_brier.append(new_brier)

test_Y_brier = np.asarray(test_Y_brier)
db_brier = {}

for ifp in set(test_df['ifp_id']):
	index = test_df.index[test_df['ifp_id'] == ifp]
	slot1 = test_df.iloc[index]['slot1'].values
	slot2 = test_df.iloc[index]['slot2'].values
	test_ideal = np.amin([slot1, slot2], axis=0)

	#pdb.set_trace()
	test_predict = test_Y_brier[index]
	db_brier[ifp] = [np.mean(slot1), np.mean(slot2), np.mean(test_ideal), np.mean(test_predict)]

with open('briers.csv', 'w') as fout:
	fout.write('slot1,slot2,ideal,predict\n')
	for ifp, row in db_brier.items():
		fout.write(str(ifp) + ',' + ','.join([str(x) for x in row]) + '\n')


briers = np.asarray(list(db_brier.values()))
print('slot1', 'slot2', 'ideal', 'predict')
print(np.mean(briers, axis=0))
#pdb.set_trace()
print(clf.score(train_X, train_Y))
print(clf.score(test_X, test_Y))
print('OK')
plt.figure(num=None, figsize=(10, 40), dpi=80)
feature_importance = clf.feature_importances_
# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, train_df.drop(columns=['ifp_id', 'date', 'slot1', 'slot2', 'answer']).columns[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
#plt.show()

plt.savefig('importance_ts.pdf')
