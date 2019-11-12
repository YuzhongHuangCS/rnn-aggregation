import pickle
import pdb
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

db = {}
name_map = {
	#'M0': 'm0',
	#'M1': 'm1',
	'M2': 'm2',
	#'M2 + feature  ': 'm3',
	#'  Transformer': 'transformer',
	'RNN': 'rnn',
	'RNN2': 'rnn2'
}

for model in ('M2', 'RNN', 'RNN2'):
	for i in [0, 1, 2]:
		filename = 'plot_data/{}_rank_db_{}.pickle'.format(model, i)
		with open(filename, 'rb') as fin:
			db_brier = pickle.load(fin)
			if model in db:
				db[model] += db_brier
			else:
				db[model] = db_brier


sns.set(style="whitegrid")
pdb.set_trace()
#df = pd.DataFrame.from_dict(db)
ax = sns.violinplot(data=[db['M2'], db['RNN'], db['RNN2']], cut=0, scale='count', inner='quartiles')
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['M2', 'RNN', 'RNN2'])
plt.ylabel('Percentile')
plt.show()
pdb.set_trace()
print('OK')
