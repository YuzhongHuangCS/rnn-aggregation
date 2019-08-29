import pickle
import pdb
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
db = {}

name_map = {
	'M0': 'm0',
	'M1': 'm1',
	'M2': 'm2',
	'M2 + feature': 'm3',
	'Transformer': 'transformer',
	'RNN': 'rnn'
}
for model in ('M0', 'M1', 'M2', 'M2 + feature', 'Transformer', 'RNN',):
	model_name = name_map[model]
	for i in range(5):
		filename = 'plot_data/{}_brier_db_{}.pickle'.format(model_name, i)
		with open(filename, 'rb') as fin:
			db_brier = pickle.load(fin)
			if model in db:
				db[model].update(db_brier)
			else:
				db[model] = db_brier

sns.set(style="whitegrid")
df = pd.DataFrame.from_dict(db)
ax = sns.violinplot(data=df, bw=0.1, cut=0, scale='count', inner='quartiles')
pdb.set_trace()
#np.quantile(list(db['m0'].values()), 0.25)
plt.show()
print('OK')
