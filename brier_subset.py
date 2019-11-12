import pickle
import pdb
import seaborn as sns
import pandas as pd
import dateutil.parser
import matplotlib.pyplot as plt
import numpy as np
db = {}

name_map = {
	'M0': 'm0',
	'M1': 'm1',
	'M2': 'm2',
	'M2 + feature  ': 'm3',
	'  Transformer': 'transformer',
	'RNN': 'rnn'
}
for model in ('M0', 'M1', 'M2', 'M2 + feature  ', '  Transformer', 'RNN',):
	model_name = name_map[model]
	for i in range(5):
		filename = 'plot_data/{}_brier_db_{}.pickle'.format(model_name, i)
		with open(filename, 'rb') as fin:
			db_brier = pickle.load(fin)
			if model in db:
				db[model].update(db_brier)
			else:
				db[model] = db_brier

df = pd.DataFrame.from_dict(db)
df['ifp_id'] = df.index
#df.to_csv('brier_csv.csv')
#human_feature = pd.read_csv('data/human_features.csv').drop_duplicates(subset=['ifp_id'], keep='last')
#ts_feature = pd.read_csv('data/ts_features.csv').drop_duplicates(subset=['ifp_id'], keep='last')
ifp_feature = pd.read_csv('data/dump_questions.csv').drop_duplicates(subset=['ifp_id'], keep='last')
#human_feature = human_feature.merge(ts_feature, how='outer', on='ifp_id')
#df = df.merge(human_feature, how='left', on='ifp_id')
#df = df.merge(ts_feature, how='left', on='ifp_id')
df = df.merge(ifp_feature, how='left', on='ifp_id')

def is_ordered(opt):
	keywords = ['Less', 'Between', 'More', 'inclusive','less', 'between', 'more']
	if len(opt) <= 2: #binary
		return False
	for o in opt:
		if any(x in o for x in keywords):
			return True

	return False

baseline_1 = []
proposed_1 = []

baseline_2 = []
proposed_2 = []
for index, row in df.iterrows():
	diff = dateutil.parser.parse(row['end_date']) - dateutil.parser.parse(row['start_date'])
	if diff.days > 42:
		baseline_1.append(row['M2'])
		proposed_1.append(row['RNN'])
	else:
		baseline_2.append(row['M2'])
		proposed_2.append(row['RNN'])


#df.to_csv('brier_csv.csv')
pdb.set_trace()
print('OK')
