import pickle
import pdb
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np
db = {}

name_map = {
	'M0': 'm0',
	'M1': 'm1',
	'M2': 'm2',
	'Transformer': 'transformer',
	'RNN': 'rnn'
}
for model in ('M0', 'M1', 'M2'):
	model_name = name_map[model]
	for i in range(5):
		filename = 'plot_data/{}_predictions_db_{}.pickle'.format(model_name, i)
		with open(filename, 'rb') as fin:
			db_pred= pickle.load(fin)
			y_score = []
			for key, value in db_pred.items():
				y_score += value
			if model in db:
				db[model] += y_score 
			else:
				db[model] = y_score

sns.set(style="whitegrid")
print(len(db['M0']))
print(len(db['M1']))
print(len(db['M2']))


def plot_calibration(db):
	fig, ax = plt.subplots()
	ax.plot([0, 1], [0, 1], "k:", label="Calibrated")
	for model in ('M0', 'M1','M2'):
		y_score = list(db[model])
		y_true = list(np.ones(len(y_score))) + list(np.zeros(len(y_score)))
		compl = [1.-x for x in y_score]
		y_score += compl
		fop, mpv = calibration_curve(y_true, y_score,  n_bins=10)
		
		ax.plot(mpv, fop, "s-",
             label="%s" % (model, ))

	ax.set_ylabel("Fraction of positives")
	ax.set_ylim([-0.05, 1.05])
	ax.legend(loc="lower right")
	ax.set_title('Calibration plots  (reliability curve)')
	plt.tight_layout()
	plt.show()


def plot_roc_curve(db):
	fig, ax = plt.subplots()
	ax.plot([0, 1], [0, 1], "k:")
	for model in ('M0', 'M1','M2'):
		y_score = list(db[model])
		y_true = list(np.ones(len(y_score))) + list(np.zeros(len(y_score)))
		compl = [1.-x for x in y_score]
		y_score += compl
		fpr, tpr, _ = roc_curve(y_true, y_score)
		roc_auc = roc_auc_score(y_true, y_score)

		plt.plot(fpr, tpr, label=' {0} (AUC = {1:0.2f})'
             ''.format(model, roc_auc))
		


	ax.set_ylabel("True Positive Rate")
	ax.set_xlabel("False Positive Rate")
	ax.set_ylim([-0.05, 1.05])
	ax.legend(loc="lower right")
	ax.set_title('ROC Curve')
	plt.tight_layout()
	plt.show()


plot_calibration(db)

plot_roc_curve(db)

