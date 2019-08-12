import os
import concurrent.futures
import pdb
human_features = ['Health/Disease', 'Macroeconomics/Finance', 'Natural Sciences/Climate',
       'Other', 'Politics/Intl Relations', 'Technology', 'entropy_b',
       'entropy_c', 'entropy_d', 'entropy_human', 'entropy_sage', 'entropy_te',
       'n_forecasts', 'n_forecasts_b', 'n_forecasts_c',
       'n_forecasts_d', 'n_forecasts_sage', 'n_forecasts_te', 'ordinal',
       'variance_b', 'variance_c', 'variance_d',
       'variance_human', 'variance_sage', 'variance_te']


executor = concurrent.futures.ThreadPoolExecutor(48)

def run_commands(cmds):
	for cmd in cmds:
		print(cmd)
		os.system(cmd)

db = {}
i = 0
for human_feature in human_features:
	for fold in range(5):
		log_dir = 'model/{}/{}'.format(human_feature.replace('/', '_').replace(' ', '_'), fold)
		os.makedirs(log_dir, exist_ok=True)
		log_path = log_dir + '/log.log'
		ckpt_path = log_dir + '/model.ckpt.index'
		if os.path.exists(ckpt_path):
			continue
		cmd = 'taskset -c {} python3 -u rnn_agg_slave.py "{}" {} > {}'.format(i, human_feature, fold, log_path)

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
