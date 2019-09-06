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
from briercompute import brier

def is_ordered(opt):
	keywords = ['Less', 'Between', 'More', 'inclusive','less', 'between', 'more']
	if len(opt) == 1: #binary
		return False
	for o in opt:
		if any(x in o for x in keywords):
			return True

	return False

db_answer = {}
for filename in ('data/dump_questions_rcta.csv', 'data/dump_questions_rctb.csv'):
	df_question = pd.read_csv(filename)
	for index, row in df_question.iterrows():
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
random.seed(2019)
random.shuffle(all_ifp_shuffle)

n_train = int(n_all*0.8)
n_test = n_all - n_train

ifp_train = all_ifp_shuffle[:n_train]
ifp_test = all_ifp_shuffle[n_train:]

N_RNN_DIM = 32

### TRAIN data
input_train = np.zeros((n_train, max_steps, 5))
target_train = np.zeros((n_train, 5))
answer_train = np.zeros(n_train, dtype=int)
is_ordered_train = np.zeros(n_train, dtype=int)
seq_length_train = np.zeros(n_train, dtype=int)
gather_index_train = np.zeros((n_train, 2), dtype=int)
num_option_mask_train = np.zeros((n_train, 5), dtype=int)

for index, ifp in enumerate(ifp_train):
	forecasts = db[ifp]

	for i, forecast in enumerate(forecasts):
		input_train[index, i] = forecast[-5:]

	answer, is_ordered = db_answer[ifp]
	target_train[index, answer] = 1
	answer_train[index] = answer
	is_ordered_train[index] = is_ordered
	seq_length_train[index] = len(forecasts)

	gather_index_train[index, :] = [index, len(forecasts)-1]

	num_options = forecasts[0][3]
	num_option_mask_train[index, :num_options] = 1


input_train[np.isnan(input_train)] = 0

### TEST data
input_test = np.zeros((n_test, max_steps, 5))
target_test = np.zeros((n_test, 5))
answer_test = np.zeros(n_test, dtype=int)
is_ordered_test = np.zeros(n_test, dtype=int)
seq_length_test = np.zeros(n_test, dtype=int)
gather_index_test = np.zeros((n_test, 2), dtype=int)
num_option_mask_test = np.zeros((n_test, 5))

for index, ifp in enumerate(ifp_test):
	forecasts = db[ifp]

	for i, forecast in enumerate(forecasts):
		input_test[index, i] = forecast[-5:]

	answer, is_ordered = db_answer[ifp]
	target_test[index, answer] = 1
	answer_test[index] = answer
	is_ordered_test[index] = is_ordered
	seq_length_test[index] = len(forecasts)

	gather_index_test[index, :] = [index, len(forecasts)-1]

	num_options = forecasts[0][3]
	num_option_mask_test[index, :num_options] = 1


input_test[np.isnan(input_test)] = 0

# Network placeholder
input_placeholder = tf.placeholder(tf.float32, [None, max_steps, 5])
target_placeholder = tf.placeholder(tf.int32, [None, 5])
is_ordered_placeholder = tf.placeholder(tf.int32, [None])
weight_placeholder = tf.placeholder(tf.float32, [None])
seq_length_placeholder = tf.placeholder(tf.float32, [None])
gather_index_placeholder = tf.placeholder(tf.int32, [None, 2])
num_option_mask_placeholder = tf.placeholder(tf.float32, [None, 5])

cell = tf.nn.rnn_cell.GRUCell(N_RNN_DIM, kernel_initializer=tf.orthogonal_initializer(), bias_initializer=tf.zeros_initializer())
state_series, _ = tf.nn.dynamic_rnn(cell, input_placeholder, sequence_length=seq_length_placeholder, dtype=tf.float32)
W1 = tf.get_variable('W1', shape=(N_RNN_DIM, 5), initializer=tf.glorot_uniform_initializer())
b1 = tf.get_variable('b1', shape=(1, 5), initializer=tf.zeros_initializer())
needed_state = tf.gather_nd(state_series, gather_index_placeholder)
prediction = tf.matmul(needed_state, W1) + b1
masked_prediction = tf.math.multiply(prediction, num_option_mask_placeholder)
prob = tf.nn.softmax(masked_prediction)

mse_loss = tf.losses.mean_squared_error(target_placeholder, prob)
train_op = tf.train.AdamOptimizer(0.005).minimize(loss=mse_loss)
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	for i in range(100):
		train_loss, train_pred, _train_step = sess.run(
			[mse_loss, prob, train_op],
				feed_dict={
					input_placeholder: input_train,
					target_placeholder: target_train,
					is_ordered_placeholder: is_ordered_train,
					#weight_placeholder: _current_cell_state,
					seq_length_placeholder: seq_length_train,
					gather_index_placeholder: gather_index_train,
					num_option_mask_placeholder: num_option_mask_train
				}
		)

		test_loss, test_pred = sess.run(
			[mse_loss, prob],
				feed_dict={
					input_placeholder: input_test,
					target_placeholder: target_test,
					is_ordered_placeholder: is_ordered_test,
					#weight_placeholder: _current_cell_state,
					seq_length_placeholder: seq_length_test,
					gather_index_placeholder: gather_index_test,
					num_option_mask_placeholder: num_option_mask_test
				}
		)

		train_brier = np.mean([brier(p, answer_train[i], is_ordered_train[i]) for i, p in enumerate(train_pred)])
		test_brier = np.mean([brier(p, answer_test[i], is_ordered_test[i]) for i, p in enumerate(test_pred)])
		print(i, train_loss, test_loss, train_brier, test_brier)


print('OK')
