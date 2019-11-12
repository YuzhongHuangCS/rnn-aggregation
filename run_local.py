import os
import concurrent.futures
import pdb
import sys

model_name = sys.argv[1]

executor = concurrent.futures.ThreadPoolExecutor(48)

def run_commands(cmds):
	for cmd in cmds:
		print(cmd)
		os.system(cmd)

db = {}
for fold_index in range(5):
	log_dir = '{}/{}/{}'.format(model_name, 'rnn_model', fold_index)
	os.makedirs(log_dir, exist_ok=True)
	log_path = log_dir + '/log.log'
	ckpt_path = log_dir + '/model.ckpt.index'
	if os.path.exists(ckpt_path):
		continue
	cmd = 'taskset -c {} python3 -u rnn_agg_train.py {} {} "{}" > {}'.format(fold_index, model_name, fold_index, 'rnn_model', log_path)
	print(cmd)
	executor.submit(os.system, cmd)

print('All task generated')
executor.shutdown()
print('All done')
