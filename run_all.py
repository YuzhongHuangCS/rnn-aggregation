import os
import concurrent.futures
import pdb
import sys

model_name = sys.argv[1]

human_feature_list = ['n_forecasts_te', 'variance_sage', 'n_forecasts_d', 'n_forecasts_sage', 'entropy_b', 'entropy_d', 'entropy_te', 'n_forecasts_b', 'entropy_c', 'Technology', 'variance_b', 'variance_d', 'Other', 'n_forecasts_c', 'stage', 'entropy_sage', 'n_forecasts', 'Politics/Intl Relations', 'Macroeconomics/Finance', 'variance_te', 'variance_c', 'variance_human', 'entropy_human', 'ordinal', 'Natural Sciences/Climate']
ts_feature_list = ['diff2_acf10', 'entropy', 'diff1_acf10', 'seas_pacf', 'linearity', 'spike', 'nonlinearity', 'diff1x_pacf5', 'e_acf10', 'series_length', 'hurst', 'ARCH.LM', 'ratio', 'seas_acf1', 'x_acf1', 'crossing_points', 'x_pacf5', 'diff1_acf1', 'trend', 'trough', 'unitroot_pp', 'diff2x_pacf5', 'x_acf10', 'nperiods', 'flat_spots', 'seasonal_period', 'peak', 'beta', 'diff2_acf1', 'lumpiness', 'e_acf1', 'skew', 'curvature', 'alpha', 'unitroot_kpss', 'seasonal_strength', 'stability']

all_feature_list = human_feature_list + ts_feature_list
executor = concurrent.futures.ThreadPoolExecutor(48)

def run_commands(cmds):
	for cmd in cmds:
		print(cmd)
		os.system(cmd)

db = {}
i = 0
for human_feature in all_feature_list:
	for fold_index in range(5):
		log_dir = '{}/{}/{}'.format(model_name, human_feature.replace('/', '_').replace(' ', '_'), fold_index)
		os.makedirs(log_dir, exist_ok=True)
		log_path = log_dir + '/log.log'
		ckpt_path = log_dir + '/model.ckpt.index'
		if os.path.exists(ckpt_path):
			continue
		cmd = 'taskset -c {} python3 -u rnn_agg.py {} {} "{}" > {}'.format(i, model_name, fold_index, human_feature, log_path)

		if i not in db:
			db[i] = [cmd]
		else:
			db[i].append(cmd)

		i = (i+1) % 48

for k, v in db.items():
	executor.submit(run_commands, v)

print('All task generated')
executor.shutdown()
print('All done')
