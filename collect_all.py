import os
import numpy as np
import sys

model_name = sys.argv[1]

human_feature_list = ['n_forecasts_te', 'variance_sage', 'n_forecasts_d', 'n_forecasts_sage', 'entropy_b', 'entropy_d', 'entropy_te', 'n_forecasts_b', 'entropy_c', 'Technology', 'variance_b', 'variance_d', 'Other', 'n_forecasts_c', 'stage', 'entropy_sage', 'n_forecasts', 'Politics/Intl Relations', 'Macroeconomics/Finance', 'Health/Disease', 'variance_te', 'variance_c', 'variance_human', 'entropy_human', 'ordinal', 'Natural Sciences/Climate', 'p_updates']
ts_feature_list = ['diff2_acf10', 'entropy', 'diff1_acf10', 'seas_pacf', 'linearity', 'spike', 'nonlinearity', 'diff1x_pacf5', 'e_acf10', 'series_length', 'hurst', 'ARCH.LM', 'ratio', 'seas_acf1', 'x_acf1', 'crossing_points', 'x_pacf5', 'diff1_acf1', 'trend', 'trough', 'unitroot_pp', 'diff2x_pacf5', 'x_acf10', 'nperiods', 'flat_spots', 'seasonal_period', 'peak', 'beta', 'diff2_acf1', 'lumpiness', 'e_acf1', 'skew', 'curvature', 'alpha', 'unitroot_kpss', 'seasonal_strength', 'stability']

all_feature_list = human_feature_list + ts_feature_list
fout = open('{}_stat.txt'.format(model_name), 'w')
for human_feature in all_feature_list:
	briers = []
	for fold in range(5):
		log_dir = '{}/{}/{}'.format(model_name, human_feature.replace('/', '_').replace(' ', '_'), fold)
		log_path = log_dir + '/log.log'
		print(log_path)

		text = open(log_path).readlines()
		assert text[-1] == 'OK\n'
		brier = float(text[-3].split()[-1])
		briers.append(brier)

	fout.write('{}: {} -> {}\n'.format(human_feature, np.mean(briers), briers))

fout.close()
