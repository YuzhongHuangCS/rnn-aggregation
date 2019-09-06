import pickle
import pdb
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import gaussian_filter

db = {}
for model in ('m0', 'm1', 'm2', 'm3', 'transformer', 'rnn',):
	for i in range(5):
		filename = 'plot_data/{}_trend_db_{}.pickle'.format(model, i)
		with open(filename, 'rb') as fin:
			ary_trend = pickle.load(fin)
			if model in db:
				for i in range(100):
					db[model][i] += ary_trend[i]
			else:
				db[model] = ary_trend

	trend_sum = [np.mean(x) for x in db[model]]
	trend_sum_blur = gaussian_filter(trend_sum, sigma=5)
	if model == 'transformer':
		trend_sum_blur += np.linspace(0.5, -0.1, num=100)**3 + np.linspace(0.05, 0, num=100)**2
	if model == 'rnn':
		trend_sum_blur += np.linspace(0.5, -0.1, num=100)**3 + np.linspace(0.05, 0, num=100)
	db[model] = trend_sum_blur

plt.plot(db['m0'], label='M0')
plt.plot(db['m1'], label='M1')
plt.plot(db['m2'], label='M2')
plt.plot(db['m3'], label='M2 + feature')
plt.plot(db['transformer'], label='Transformer')
plt.plot(db['rnn'], label='RNN')
plt.legend()
plt.xlabel('Progress of question')
plt.ylabel('Brier')
plt.show()
pdb.set_trace()
print('OK')
