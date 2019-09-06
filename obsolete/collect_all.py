import os
import numpy as np

human_feature_list = ['Macroeconomics/Finance', 'Natural Sciences/Climate',
	'Other', 'Politics/Intl Relations', 'Technology', 'entropy_b',
	'entropy_c', 'entropy_d', 'entropy_human', 'entropy_sage', 'entropy_te',
	'n_forecasts', 'n_forecasts_b', 'n_forecasts_c',
	'n_forecasts_d', 'n_forecasts_sage', 'n_forecasts_te', 'ordinal',
	'variance_b', 'variance_c', 'variance_d',
	'variance_human', 'variance_sage', 'variance_te']

ts_feature_list = ['x_acf1', 'x_acf10', 'diff1_acf1', 'diff1_acf10',
   'diff2_acf1', 'diff2_acf10', 'seas_acf1', 'ARCH.LM', 'crossing_points',
   'entropy', 'flat_spots', 'arch_acf', 'garch_acf', 'arch_r2', 'garch_r2',
   'alpha', 'beta', 'hurst', 'lumpiness', 'nonlinearity', 'x_pacf5',
   'diff1x_pacf5', 'diff2x_pacf5', 'seas_pacf', 'nperiods',
   'seasonal_period', 'trend', 'spike', 'linearity', 'curvature', 'e_acf1',
   'e_acf10', 'seasonal_strength', 'peak', 'trough', 'stability',
   'hw_alpha', 'hw_beta', 'hw_gamma', 'unitroot_kpss', 'unitroot_pp',
   'series_length', 'ratio', 'skew']

all_feature_list = human_feature_list + ts_feature_list
fout = open('stat.log', 'w')
for human_feature in all_feature_list:
	briers = []
	for fold in range(5):
		log_dir = 'model/{}/{}'.format(human_feature.replace('/', '_').replace(' ', '_'), fold)
		log_path = log_dir + '/log.log'
		text = open(log_path).readlines()
		print(log_path)
		assert text[-1] == 'OK\n'
		brier = float(text[-2].split()[-1])
		briers.append(brier)

	fout.write('{}: {} -> {}\n'.format(human_feature, np.mean(briers), briers))

fout.close()
