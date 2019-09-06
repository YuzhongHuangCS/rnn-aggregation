import pandas as pd
import pdb
import dateutil.parser
import numpy as np
from collections import OrderedDict
import pickle
import sklearn.model_selection
import sklearn.decomposition
import sklearn.cluster
import scipy.stats

df = pd.read_csv('data/human.csv')
#df.fillna(0, inplace=True)
#pdb.set_trace()
#np.unique(df[df['ifp_id'].isin(ifp_all)]['user_id']).shape
db = OrderedDict()

for index, row in df.iterrows():
	date = dateutil.parser.parse(row['date']).strftime('%Y-%m-%d')
	user_id = row['user_id']
	ifp_id = row['ifp_id']
	num_options = row['num_options']
	option_1 = row['option_1']
	option_2 = row['option_2']
	option_3 = row['option_3']
	option_4 = row['option_4']
	option_5 = row['option_5']

	if ifp_id not in db:
		db[ifp_id] = {}

	if date not in db[ifp_id]:
		db[ifp_id][date] = {}

	db[ifp_id][date][user_id] = [date,user_id,ifp_id,num_options,option_1,option_2,option_3,option_4,option_5]

MAX_CLUSTER = 4
human_ary = []

for ifp_id, dates in db.items():
	for date, user_ids in dates.items():
		num_options = next(iter(user_ids.values()))[3]
		if num_options == 1:
			num_options = 2

		forecasts = np.asarray([x[-5:] for x in user_ids.values()])
		forecast = np.mean(forecasts, axis=0) / 100

		n_clusters = min(len(forecasts), MAX_CLUSTER)
		kmeans = sklearn.cluster.KMeans(n_clusters=n_clusters).fit(forecasts[:, :num_options])
		pivots = [(scipy.stats.entropy(x), x / 100) for x in kmeans.cluster_centers_]
		pivots.sort(key=lambda tup: tup[0])

		forecast_ary = [np.pad(x[1], (0, 5-len(x[1])), 'constant', constant_values=np.nan) for x in pivots]
		if MAX_CLUSTER > 1:
			while len(forecast_ary) < MAX_CLUSTER+1:
				forecast_ary.insert(0, forecast)


		human_ary.append([date, ifp_id, num_options, *np.asarray(forecast_ary).flatten()])

human_df = pd.DataFrame(human_ary).rename(columns={0: 'date', 1: 'ifp_id', 2: 'num_options', 3: 'h_1', 4: 'h_2', 5: 'h_3', 6: 'h_4', 7: 'h_5', 8: 'h1_1', 9: 'h1_2', 10: 'h1_3', 11: 'h1_4', 12: 'h1_5', 13: 'h2_1', 14: 'h2_2', 15: 'h2_3', 16: 'h2_4', 17: 'h2_5', 18: 'h3_1', 19: 'h3_2', 20: 'h3_3', 21: 'h3_4', 22: 'h3_5', 23: 'h4_1', 24: 'h4_2', 25: 'h4_3', 26: 'h4_4', 27: 'h4_5'})
machine_all_df = pd.read_csv('data/machine_all.csv')
#machine_all_df.fillna(0, inplace=True)

machine_arima_df = machine_all_df[machine_all_df['machine_model'] == 'Auto ARIMA'] \
	.drop_duplicates(subset=['date', 'ifp_id'], keep='last') \
	.reset_index() \
	.drop(columns=['index', 'machine_model', 'num_options']) \
	.rename(columns={'option_1': 'arima_1', 'option_2': 'arima_2', 'option_3': 'arima_3', 'option_4': 'arima_4', 'option_5': 'arima_5'})

machine_m4_meta_df = machine_all_df[machine_all_df['machine_model'] == 'M4-Meta'] \
	.drop_duplicates(subset=['date', 'ifp_id'], keep='last') \
	.reset_index() \
	.drop(columns=['index', 'machine_model', 'num_options']) \
	.rename(columns={'option_1': 'm4_1', 'option_2': 'm4_2', 'option_3': 'm4_3', 'option_4': 'm4_4', 'option_5': 'm4_5'})

machine_arw_df = machine_all_df[machine_all_df['machine_model'] == 'Arithmetic RW'] \
	.drop_duplicates(subset=['date', 'ifp_id'], keep='last') \
	.reset_index() \
	.drop(columns=['index', 'machine_model', 'num_options']) \
	.rename(columns={'option_1': 'arw_1', 'option_2': 'arw_2', 'option_3': 'arw_3', 'option_4': 'arw_4', 'option_5': 'arw_5'})

machine_dsholt_df = machine_all_df[machine_all_df['machine_model'] == 'DS-Holt'] \
	.drop_duplicates(subset=['date', 'ifp_id'], keep='last') \
	.reset_index() \
	.drop(columns=['index', 'machine_model', 'num_options']) \
	.rename(columns={'option_1': 'dsholt_1', 'option_2': 'dsholt_2', 'option_3': 'dsholt_3', 'option_4': 'dsholt_4', 'option_5': 'dsholt_5'})

machine_dsholtd_df = machine_all_df[machine_all_df['machine_model'] == 'DS-Holt-damped'] \
	.drop_duplicates(subset=['date', 'ifp_id'], keep='last') \
	.reset_index() \
	.drop(columns=['index', 'machine_model', 'num_options']) \
	.rename(columns={'option_1': 'dsholtd_1', 'option_2': 'dsholtd_2', 'option_3': 'dsholtd_3', 'option_4': 'dsholtd_4', 'option_5': 'dsholtd_5'})

machine_dsrw_df = machine_all_df[machine_all_df['machine_model'] == 'DS-RW'] \
	.drop_duplicates(subset=['date', 'ifp_id'], keep='last') \
	.reset_index() \
	.drop(columns=['index', 'machine_model', 'num_options']) \
	.rename(columns={'option_1': 'dsrw_1', 'option_2': 'dsrw_2', 'option_3': 'dsrw_3', 'option_4': 'dsrw_4', 'option_5': 'dsrw_5'})

machine_dsses_df = machine_all_df[machine_all_df['machine_model'] == 'DS-SES'] \
	.drop_duplicates(subset=['date', 'ifp_id'], keep='last') \
	.reset_index() \
	.drop(columns=['index', 'machine_model', 'num_options']) \
	.rename(columns={'option_1': 'dsses_1', 'option_2': 'dsses_2', 'option_3': 'dsses_3', 'option_4': 'dsses_4', 'option_5': 'dsses_5'})

machine_ets_df = machine_all_df[machine_all_df['machine_model'] == 'ETS'] \
	.drop_duplicates(subset=['date', 'ifp_id'], keep='last') \
	.reset_index() \
	.drop(columns=['index', 'machine_model', 'num_options']) \
	.rename(columns={'option_1': 'ets_1', 'option_2': 'ets_2', 'option_3': 'ets_3', 'option_4': 'ets_4', 'option_5': 'ets_5'})

machine_grw_df = machine_all_df[machine_all_df['machine_model'] == 'Geometric RW'] \
	.drop_duplicates(subset=['date', 'ifp_id'], keep='last') \
	.reset_index() \
	.drop(columns=['index', 'machine_model', 'num_options']) \
	.rename(columns={'option_1': 'grw_1', 'option_2': 'grw_2', 'option_3': 'grw_3', 'option_4': 'grw_4', 'option_5': 'grw_5'})

machine_m4c_df = machine_all_df[machine_all_df['machine_model'] == 'M4-Comb'] \
	.drop_duplicates(subset=['date', 'ifp_id'], keep='last') \
	.reset_index() \
	.drop(columns=['index', 'machine_model', 'num_options']) \
	.rename(columns={'option_1': 'm4c_1', 'option_2': 'm4c_2', 'option_3': 'm4c_3', 'option_4': 'm4c_4', 'option_5': 'm4c_5'})

machine_mean_df = machine_all_df[machine_all_df['machine_model'] == 'Mean'] \
	.drop_duplicates(subset=['date', 'ifp_id'], keep='last') \
	.reset_index() \
	.drop(columns=['index', 'machine_model', 'num_options']) \
	.rename(columns={'option_1': 'mean_1', 'option_2': 'mean_2', 'option_3': 'mean_3', 'option_4': 'mean_4', 'option_5': 'mean_5'})

machine_nnetar_df = machine_all_df[machine_all_df['machine_model'] == 'NNETAR'] \
	.drop_duplicates(subset=['date', 'ifp_id'], keep='last') \
	.reset_index() \
	.drop(columns=['index', 'machine_model', 'num_options']) \
	.rename(columns={'option_1': 'nnetar_1', 'option_2': 'nnetar_2', 'option_3': 'nnetar_3', 'option_4': 'nnetar_4', 'option_5': 'nnetar_5'})

machine_rnn_df = machine_all_df[machine_all_df['machine_model'] == 'RNN'] \
	.drop_duplicates(subset=['date', 'ifp_id'], keep='last') \
	.reset_index() \
	.drop(columns=['index', 'machine_model', 'num_options']) \
	.rename(columns={'option_1': 'rnn_1', 'option_2': 'rnn_2', 'option_3': 'rnn_3', 'option_4': 'rnn_4', 'option_5': 'rnn_5'})

machine_rw_df = machine_all_df[machine_all_df['machine_model'] == 'RW'] \
	.drop_duplicates(subset=['date', 'ifp_id'], keep='last') \
	.reset_index() \
	.drop(columns=['index', 'machine_model', 'num_options']) \
	.rename(columns={'option_1': 'rw_1', 'option_2': 'rw_2', 'option_3': 'rw_3', 'option_4': 'rw_4', 'option_5': 'rw_5'})

machine_rwd_df = machine_all_df[machine_all_df['machine_model'] == 'RW-DRIFT'] \
	.drop_duplicates(subset=['date', 'ifp_id'], keep='last') \
	.reset_index() \
	.drop(columns=['index', 'machine_model', 'num_options']) \
	.rename(columns={'option_1': 'rwd_1', 'option_2': 'rwd_2', 'option_3': 'rwd_3', 'option_4': 'rwd_4', 'option_5': 'rwd_5'})

machine_rws_df = machine_all_df[machine_all_df['machine_model'] == 'RW-SEAS'] \
	.drop_duplicates(subset=['date', 'ifp_id'], keep='last') \
	.reset_index() \
	.drop(columns=['index', 'machine_model', 'num_options']) \
	.rename(columns={'option_1': 'rws_1', 'option_2': 'rws_2', 'option_3': 'rws_3', 'option_4': 'rws_4', 'option_5': 'rws_5'})

#machine_sim_df = machine_all_df[machine_all_df['machine_model'] == 'Similarity Tool'] \
#	.drop_duplicates(subset=['date', 'ifp_id'], keep='last') \
#	.reset_index() \
#	.drop(columns=['index', 'machine_model', 'num_options']) \
#	.rename(columns={'option_1': 'sim_1', 'option_2': 'sim_2', 'option_3': 'sim_3', 'option_4': 'sim_4', 'option_5': 'sim_5'})

machine_sltmar_df = machine_all_df[machine_all_df['machine_model'] == 'STLM-AR'] \
	.drop_duplicates(subset=['date', 'ifp_id'], keep='last') \
	.reset_index() \
	.drop(columns=['index', 'machine_model', 'num_options']) \
	.rename(columns={'option_1': 'sltmar_1', 'option_2': 'sltmar_2', 'option_3': 'sltmar_3', 'option_4': 'sltmar_4', 'option_5': 'sltmar_5'})

machine_tbats_df = machine_all_df[machine_all_df['machine_model'] == 'TBATS'] \
	.drop_duplicates(subset=['date', 'ifp_id'], keep='last') \
	.reset_index() \
	.drop(columns=['index', 'machine_model', 'num_options']) \
	.rename(columns={'option_1': 'tbats_1', 'option_2': 'tbats_2', 'option_3': 'tbats_3', 'option_4': 'tbats_4', 'option_5': 'tbats_5'})

machine_theta_df = machine_all_df[machine_all_df['machine_model'] == 'THETA'] \
	.drop_duplicates(subset=['date', 'ifp_id'], keep='last') \
	.reset_index() \
	.drop(columns=['index', 'machine_model', 'num_options']) \
	.rename(columns={'option_1': 'theta_1', 'option_2': 'theta_2', 'option_3': 'theta_3', 'option_4': 'theta_4', 'option_5': 'theta_5'})

machine_df = pd.merge(machine_arima_df, machine_m4_meta_df, how='outer', on=['date', 'ifp_id'], validate='one_to_one')
machine_df = pd.merge(machine_df, machine_arw_df, how='outer', on=['date', 'ifp_id'], validate='one_to_one')
machine_df = pd.merge(machine_df, machine_dsholt_df, how='outer', on=['date', 'ifp_id'], validate='one_to_one')
machine_df = pd.merge(machine_df, machine_dsholtd_df, how='outer', on=['date', 'ifp_id'], validate='one_to_one')
machine_df = pd.merge(machine_df, machine_dsrw_df, how='outer', on=['date', 'ifp_id'], validate='one_to_one')
machine_df = pd.merge(machine_df, machine_dsses_df, how='outer', on=['date', 'ifp_id'], validate='one_to_one')
machine_df = pd.merge(machine_df, machine_ets_df, how='outer', on=['date', 'ifp_id'], validate='one_to_one')
machine_df = pd.merge(machine_df, machine_grw_df, how='outer', on=['date', 'ifp_id'], validate='one_to_one')
machine_df = pd.merge(machine_df, machine_m4c_df, how='outer', on=['date', 'ifp_id'], validate='one_to_one')
machine_df = pd.merge(machine_df, machine_mean_df, how='outer', on=['date', 'ifp_id'], validate='one_to_one')
machine_df = pd.merge(machine_df, machine_nnetar_df, how='outer', on=['date', 'ifp_id'], validate='one_to_one')
machine_df = pd.merge(machine_df, machine_rnn_df, how='outer', on=['date', 'ifp_id'], validate='one_to_one')
machine_df = pd.merge(machine_df, machine_rw_df, how='outer', on=['date', 'ifp_id'], validate='one_to_one')
machine_df = pd.merge(machine_df, machine_rwd_df, how='outer', on=['date', 'ifp_id'], validate='one_to_one')
machine_df = pd.merge(machine_df, machine_rws_df, how='outer', on=['date', 'ifp_id'], validate='one_to_one')
#machine_df = pd.merge(machine_df, machine_sim_df, how='outer', on=['date', 'ifp_id'], validate='one_to_one')
machine_df = pd.merge(machine_df, machine_sltmar_df, how='outer', on=['date', 'ifp_id'], validate='one_to_one')
machine_df = pd.merge(machine_df, machine_tbats_df, how='outer', on=['date', 'ifp_id'], validate='one_to_one')
machine_df = pd.merge(machine_df, machine_theta_df, how='outer', on=['date', 'ifp_id'], validate='one_to_one')

combine_df = pd.merge(human_df, machine_df, how='inner', on=['date', 'ifp_id'], validate='one_to_one')
def is_ordered(opt):
	keywords = ['Less', 'Between', 'More', 'inclusive','less', 'between', 'more']
	if len(opt) == 1: #binary
		return False
	for o in opt:
		if any(x in o for x in keywords):
			return True

	return False

db_answer = []
df_question_rcta = pd.read_csv('data/dump_questions_rcta.csv')
for index, row in df_question_rcta.iterrows():
	if row['is_resolved']:
		ifp_id = row['ifp_id']
		resolution = row['resolution']
		options = row.tolist()[-5:]

		clean_options = [x for x in options if type(x) == str]
		try:
			answer = options.index(resolution)
			db_answer.append([ifp_id, answer, is_ordered(clean_options)])
		except ValueError as e:
			pdb.set_trace()
			print(e)

df_question_rctb = pd.read_csv('data/dump_questions_rctb.csv')
for index, row in df_question_rctb.iterrows():
	if row['is_resolved']:
		ifp_id = row['ifp_id']
		resolution = row['resolution']
		options = row.tolist()[-5:]

		clean_options = [x for x in options if type(x) == str]
		try:
			answer = options.index(resolution)
			db_answer.append([ifp_id, answer, is_ordered(clean_options)])
		except ValueError as e:
			pdb.set_trace()

			print(e)

human_feature = pd.read_csv('data/human_features.csv').drop_duplicates(subset=['date', 'ifp_id'], keep='last').reset_index().drop(columns=['index'])
ts_feature = pd.read_csv('data/ts_features.csv')

combine_df = pd.merge(combine_df, human_feature, how='left', on=['ifp_id', 'date'], validate='one_to_one')
combine_df = pd.merge(combine_df, ts_feature, how='left', on=['ifp_id', 'date'], validate='one_to_one')

answer_df = pd.DataFrame(db_answer).rename(columns={0: 'ifp_id', 1: 'answer', 2: 'ordered'})
combine_df = pd.merge(combine_df, answer_df, how='inner', on=['ifp_id'], validate='many_to_one')
#combine_df.fillna(0, inplace=True)

ifp_all = np.unique(combine_df['ifp_id'])

'''
np.random.shuffle(ifp_all)
n_train = int(len(ifp_all) * 0.8)
ifp_train = set(ifp_all[:n_train])
ifp_test = set(ifp_all[n_train:])

train_df = combine_df[combine_df['ifp_id'].isin(ifp_train)].reset_index().drop(columns=['index'])
test_df = combine_df[combine_df['ifp_id'].isin(ifp_test)].reset_index().drop(columns=['index'])
'''
'''

kf = sklearn.model_selection.KFold(shuffle=True, n_splits=5, random_state=2019)
counter = 0
for train_index, test_index in kf.split(ifp_all):
	ifp_train = set(ifp_all[train_index])
	ifp_test = set(ifp_all[test_index])
	train_df = combine_df[combine_df['ifp_id'].isin(ifp_train)].reset_index().drop(columns=['index'])
	test_df = combine_df[combine_df['ifp_id'].isin(ifp_test)].reset_index().drop(columns=['index'])

	with open('dataset_{}.pickle'.format(counter), 'wb') as f:
		pickle.dump([train_df, test_df], f, pickle.HIGHEST_PROTOCOL)

	counter += 1
'''
all_df = combine_df.reset_index().drop(columns=['index'])
with open('dataset_all.pickle'.format(0), 'wb') as f:
	pickle.dump([ifp_all, all_df], f, pickle.HIGHEST_PROTOCOL)

print('OK')
