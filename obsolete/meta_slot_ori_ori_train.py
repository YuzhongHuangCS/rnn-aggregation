import dateparser
import pdb
import pickle
import os
import pandas as pd
import sklearn
import sklearn.linear_model
import sklearn.ensemble
import sklearn.svm
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt


slot_1_name = '07_BCD_T_M11b'
slot_2_name = 'M_extreme2'

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
					train_df_s1.append([ifp, date, brier, max(prob)])

				if slot == slot_2_name:
					train_df_s2.append([ifp, date, brier])

			if ifp in ifp_test:
				if slot == slot_1_name:
					test_df_s1.append([ifp, date, brier, max(prob)])

				if slot == slot_2_name:
					test_df_s2.append([ifp, date, brier])

train_df_s1 = pd.DataFrame(train_df_s1).rename(columns={0: 'ifp_id', 1: 'date', 2: 'slot1', 3: 'max_prob'})
train_df_s2 = pd.DataFrame(train_df_s2).rename(columns={0: 'ifp_id', 1: 'date', 2: 'slot2'})
train_df = pd.merge(train_df_s1, train_df_s2, how='inner', on=['ifp_id', 'date'])
test_df_s1 = pd.DataFrame(test_df_s1).rename(columns={0: 'ifp_id', 1: 'date', 2: 'slot1', 3: 'max_prob'})
test_df_s2 = pd.DataFrame(test_df_s2).rename(columns={0: 'ifp_id', 1: 'date', 2: 'slot2'})
test_df = pd.merge(test_df_s1, test_df_s2, how='inner', on=['ifp_id', 'date'])

human_feature = pd.read_csv('human_features.csv')
ts_feature = pd.read_csv('ts_features_rctb.csv')

train_df = pd.merge(train_df, human_feature, how='inner', on=['ifp_id', 'date'])
train_df = pd.merge(train_df, ts_feature, how='inner', on=['ifp_id', 'date'])

test_df = pd.merge(test_df, human_feature, how='inner', on=['ifp_id', 'date'])
test_df = pd.merge(test_df, ts_feature, how='inner', on=['ifp_id', 'date'])

train_X = train_df.drop(columns=['ifp_id', 'date', 'slot1', 'slot2']).values
train_Y = (train_df['slot1'] - train_df['slot2']).astype(float)


with open('dataset.csv', 'w') as fout:
	fout.write(','.join(train_df.columns) + ',split\n')
	for index, row in train_df.iterrows():
		fout.write(','.join([str(x) for x in row]) + ',1\n')

	for index, row in test_df.iterrows():
		fout.write(','.join([str(x) for x in row]) + ',0\n')

#exit()
#reg = sklearn.linear_model.ElasticNet(alpha=0.78, l1_ratio=0.5)
#reg = sklearn.svm.SVR(C=10)
reg = sklearn.ensemble.GradientBoostingRegressor(loss='ls', n_estimators=5000)
#reg = sklearn.linear_model.LinearRegression()
reg.fit(train_X, train_Y)
#pdb.set_trace()
#print(reg.coef_)

#test_X = test_df.drop(columns=['ifp_id', 'date', 'slot1', 'slot2']).values
#test_Y = (test_df['slot1'] - test_df['slot2']).astype(float)
test_df = train_df
test_X = train_X
test_Y = train_Y
test_Y_hat = reg.predict(test_X)
test_hat_avg = []
for i, v in enumerate(test_Y_hat):
	if v > 0.05:
		test_hat_avg.append(test_df['slot2'][i])
	else:
		test_hat_avg.append(test_df['slot1'][i])

test_hat_avg = np.asarray(test_hat_avg)
db_brier = {}

for ifp in test_df['ifp_id']:
	index = test_df.index[test_df['ifp_id'] == ifp]
	slot1 = test_df.iloc[index]['slot1'].values
	slot2 = test_df.iloc[index]['slot2'].values
	test_ideal = np.amin([slot1, slot2], axis=0)
	test_predict = test_hat_avg[index]
	db_brier[ifp] = [np.mean(slot1), np.mean(slot2), np.mean(test_ideal), np.mean(test_predict)]

briers = np.asarray(list(db_brier.values()))
print('slot1', 'slot2', 'ideal', 'predict')
print(np.mean(briers, axis=0))
#pdb.set_trace()
print('OK')

'''
plt.figure(num=None, figsize=(10, 40), dpi=80)
feature_importance = reg.feature_importances_
# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, train_df.drop(columns=['ifp_id', 'date', 'slot1', 'slot2']).columns[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
#plt.show()
plt.savefig('some.pdf')
'''
