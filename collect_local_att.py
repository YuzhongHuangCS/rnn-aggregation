import os
import numpy as np
import sys

model_name = sys.argv[1]


fout = open('{}_stat.txt'.format(model_name), 'w')
briers = []
for fold in range(5):
	log_dir = '{}/{}/{}'.format(model_name, 'att_model', fold)
	log_path = log_dir + '/log.log'
	print(log_path)

	text = open(log_path).readlines()
	assert text[-1] == 'OK\n'
	brier = float(text[-3].split()[-1])
	briers.append(brier)

fout.write('{}: {} -> {}\n'.format('att_model', np.mean(briers), briers))
fout.close()
