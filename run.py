import os

for i in range(10):
	cmd = 'python -u rnn_agg_en2.py > log_noemb/{}.txt'.format(i)
	print(cmd)
	os.system(cmd)
