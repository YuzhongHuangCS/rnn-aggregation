import pickle
import pdb
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

db = {}
name_map = {
	'M0': 'm0',
	'M1': 'm1',
	'M2': 'm2',
	'M2 + feature': 'm3',
	'Transformer': 'transformer',
	'RNN': 'rnn'
}

for model in ('m0', 'm1', 'm2', 'm3', 'transformer', 'rnn',):
	for i in range(5):
		filename = 'plot_data/{}_rank_db_{}.pickle'.format(model, i)
		with open(filename, 'rb') as fin:
			db_brier = pickle.load(fin)
			if model in db:
				db[model] += db_brier
			else:
				db[model] = db_brier


sns.set(style="whitegrid")
#pdb.set_trace()
#df = pd.DataFrame.from_dict(db)
ax = sns.violinplot(data=[db['m0'], db['m1'], db['m2'], db['m3'], db['transformer'], db['rnn']], cut=0, scale='count', inner='quartiles')
ax.set_xticks([0, 1, 2, 3, 4, 5])
ax.set_xticklabels(['M0', 'M1', 'M2', 'M2 + feature', 'Transformer', 'RNN'])
plt.show()
pdb.set_trace()
print('OK')
