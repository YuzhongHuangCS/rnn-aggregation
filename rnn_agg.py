import os

# uncomment to force CPU training
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# optimize CPU performance
os.environ['KMP_BLOCKTIME'] = '0'
os.environ['KMP_AFFINITY'] = 'granularity=fine,verbose,compact,1,0'


import pandas as pd
import pdb
import dateutil.parser
import numpy as np
from collections import OrderedDict
import pickle
import sklearn.model_selection
import sklearn.decomposition
import sklearn.cluster
import scipy.stats
import tensorflow as tf
import random
import copy

def is_ordered(opt):
	keywords = ['Less', 'Between', 'More', 'inclusive','less', 'between', 'more']
	if len(opt) == 1: #binary
		return False
	for o in opt:
		if any(x in o for x in keywords):
			return True

	return False

db_answer = {}
df_question_rcta = pd.read_csv('data/dump_questions_rcta.csv')
for index, row in df_question_rcta.iterrows():
	if row['is_resolved']:
		ifp_id = row['ifp_id']
		resolution = row['resolution']
		options = row.tolist()[-5:]

		clean_options = [x for x in options if type(x) == str]
		try:
			answer = options.index(resolution)
			if ifp_id in db_answer:
				pdb.set_trace()
				print(ifp_id)
			else:
				db_answer[ifp_id] = [answer, is_ordered(clean_options)]
		except ValueError as e:
			pdb.set_trace()
			print(e)

df_question_rctb = pd.read_csv('data/dump_questions_rctb.csv')
for index, row in df_question_rctb.iterrows():
	if row['is_resolved']:
		ifp_id = row['ifp_id']
		resolution = row['resolution']
		options = row.tolist()[-5:]

		clean_options = [x for x in options if type(x) == str]
		try:
			answer = options.index(resolution)
			if ifp_id in db_answer:
				pdb.set_trace()
				print(ifp_id)
			else:
				db_answer[ifp_id] = [answer, is_ordered(clean_options)]
		except ValueError as e:
			pdb.set_trace()

			print(e)

df = pd.read_csv('data/human.csv')
#df.fillna(0, inplace=True)
#pdb.set_trace()
#np.unique(df[df['ifp_id'].isin(ifp_all)]['user_id']).shape
db = OrderedDict()

for index, row in df.iterrows():
	date = row['date']
	user_id = row['user_id']
	ifp_id = row['ifp_id']

	if ifp_id not in db_answer:
		continue

	num_options = row['num_options']
	option_1 = row['option_1']
	option_2 = row['option_2']
	option_3 = row['option_3']
	option_4 = row['option_4']
	option_5 = row['option_5']

	if num_options == 1:
		num_options = 2

	if ifp_id not in db:
		db[ifp_id] = []
	else:
		if dateutil.parser.parse(date) < dateutil.parser.parse(db[ifp_id][-1][0]):
			if (dateutil.parser.parse(db[ifp_id][-1][0])-dateutil.parser.parse(date)).total_seconds() > 1:
				print('Date not in ascending order', date, db[ifp_id][-1][0])

	db[ifp_id].append([date,user_id,ifp_id,num_options,option_1,option_2,option_3,option_4,option_5])

max_steps = max([len(v) for k, v in db.items()])
n_all = len(db)

all_ifp = list(db.keys())
all_ifp_shuffle = copy.deepcopy(all_ifp)
random.shuffle(all_ifp_shuffle)

n_train = int(n_all*0.8)
n_test = n_all - n_train

ifp_train = all_ifp_shuffle[:n_train]
ifp_test = all_ifp_shuffle[n_train:]

N_RNN_DIM = 32
data_train = np.zeros((n_train, max_steps, 5))
n_opt_ary_train = np.zeros(n_train, dtype=int)
seq_length_ary_train = np.zeros(n_train, dtype=int)
answer_ary_train = []

for index, ifp in enumerate(ifp_train):
	forecasts = db[ifp]

	n_opt_ary_train[index] = forecasts[0][3]
	seq_length_ary_train[index] = len(forecasts)
	answer_ary_train.append(db_answer[ifp])
	for i, forecast in enumerate(forecasts):
		data_train[index, i] = forecast[-5:]

data_train[np.isnan(data_train)] = 0


gather_index_ary_train = []
for i, l in enumerate(seq_length_ary_train):
	gather_index_ary_train.append([i, l-1])

ifp_count = n_train
batchX_placeholder = tf.placeholder(tf.float32, [None, max_steps, 5])
cell = tf.nn.rnn_cell.GRUCell(N_RNN_DIM, kernel_initializer=tf.orthogonal_initializer(), bias_initializer=tf.zeros_initializer())
state_series, _ = tf.nn.dynamic_rnn(cell, batchX_placeholder, dtype=tf.float32)

W1 = tf.get_variable('W1', shape=(N_RNN_DIM, 5), initializer=tf.glorot_uniform_initializer())
b1 = tf.get_variable('b1', shape=(1, 5), initializer=tf.zeros_initializer())

needed_state_train = tf.gather_nd(state_series, gather_index_ary_train)
prediction = tf.matmul(needed_state_train, W1) + b1
# TODO: set additional value to 0
num_option_mask = np.zeros((ifp_count, 5))
num_option_ary = n_opt_ary_train
for i, c in enumerate(num_option_ary):
	num_option_mask[i, :c] = 1

num_option_mask_tensor = tf.constant(num_option_mask, dtype=tf.float32)


prob = tf.nn.softmax(tf.math.multiply(prediction, num_option_mask_tensor))

target_prob = np.zeros((ifp_count, 5))
answer_ary = answer_ary_train
for i, a in enumerate(answer_ary):
	target_prob[i, a[0]] = 1

batchY_placeholder = tf.placeholder(tf.float32, [ifp_count, 5])

mse_loss = tf.losses.mean_squared_error(batchY_placeholder, prob)
train_op = tf.train.AdamOptimizer(0.005).minimize(loss=mse_loss)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(100):
       train_loss, _train_step = sess.run(
        [mse_loss, train_op],
            feed_dict={
                batchX_placeholder: data_train,
                batchY_placeholder: target_prob,
            }
        )
       print(i, train_loss)
print('OK')
