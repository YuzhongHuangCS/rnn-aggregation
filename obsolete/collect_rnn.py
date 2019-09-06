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
ifp_count = len(db)

data = np.zeros((ifp_count, max_steps, 5))
num_option_ary = np.zeros(ifp_count, dtype=int)
length_ary = np.zeros(ifp_count, dtype=int)
answer_ary = []

for index, pair in enumerate(db.items()):
	ifp, forecasts = pair
	num_option_ary[index] = forecasts[0][3]
	length_ary[index] = len(forecasts)
	answer_ary.append(db_answer[ifp])
	for i, forecast in enumerate(forecasts):
		data[index, i] = forecast[-5:]

data[np.isnan(data)] = 0

batchX_placeholder = tf.placeholder(tf.float32, [ifp_count, max_steps, 5])
cell = tf.nn.rnn_cell.GRUCell(32, kernel_initializer=tf.orthogonal_initializer(), bias_initializer=tf.zeros_initializer())
cell_state = cell.zero_state(ifp_count, dtype=tf.float32)
states_series, current_state = tf.nn.dynamic_rnn(cell, batchX_placeholder, initial_state=cell_state, parallel_iterations=32, sequence_length=length_ary)

W1 = tf.get_variable('W1', shape=(32, 5), initializer=tf.glorot_uniform_initializer())
b1 = tf.get_variable('b1', shape=(1, 5), initializer=tf.zeros_initializer())


index = []
for i, l in enumerate(length_ary):
	index.append([i, l-1])

needed_state = tf.gather_nd(states_series, index)
prediction = tf.matmul(needed_state, W1) + b1
# TODO: set additional value to 0
num_option_mask = np.zeros((ifp_count, 5))
for i, c in enumerate(num_option_ary):
	num_option_mask[i, :c] = 1

num_option_mask_tensor = tf.constant(num_option_mask, dtype=tf.float32)


prob = tf.nn.softmax(tf.math.multiply(prediction, num_option_mask_tensor))

target_prob = np.zeros((ifp_count, 5))
for i, a in enumerate(answer_ary):
	target_prob[i, a[0]] = 1

batchY_placeholder = tf.placeholder(tf.float32, [ifp_count, 5])

mse_loss = tf.losses.mean_squared_error(batchY_placeholder, prob)
_current_cell_state = np.zeros((ifp_count, 32), dtype=np.float32)
train_op = tf.train.AdamOptimizer(0.005).minimize(loss=mse_loss)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(100):
       train_loss, _train_step = sess.run(
        [mse_loss, train_op],
            feed_dict={
                batchX_placeholder: data,
                batchY_placeholder: target_prob,
                cell_state: _current_cell_state,
            }
        )
       print(i, train_loss)
print('OK')
