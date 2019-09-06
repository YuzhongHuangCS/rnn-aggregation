import tensorflow as tf
import pickle
import pandas as pd
import numpy as np
import pdb
from briercompute import brier as brier_compute
import os

# uncomment to force CPU training
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# optimize CPU performance
#os.environ['KMP_BLOCKTIME'] = '0'
#os.environ['KMP_AFFINITY'] = 'granularity=fine,verbose,compact,1,0'


with open('dataset_0.pickle', 'rb') as f:
	train_df, test_df = pickle.load(f)

train_df = train_df.sample(frac=1)
n_valid = int(len(train_df)*0.8)
n_valid = 0
valid_df = train_df.iloc[:n_valid]
train_df = train_df.iloc[n_valid:]

train_X = train_df.drop(columns=['ifp_id', 'date', 'slot1', 'slot2', 'answer', 'ordered']).values
train_Y = np.zeros((len(train_X), 5))
for i, v in enumerate(train_df['answer'].values):
	train_Y[i, v] = 1
train_Z = train_df['ordered'].values.astype(int)

train_count_db = {}
for ifp in set(train_df['ifp_id']):
	index = train_df.index[train_df['ifp_id'] == ifp]
	train_count_db[ifp] = 1.0/len(index)

train_W = []
for index, row in train_df.iterrows():
	train_W.append(train_count_db[row['ifp_id']])

train_W = np.asarray(train_W)
train_W /= len(set(train_df['ifp_id']))
train_W *= len(train_df)

valid_X = valid_df.drop(columns=['ifp_id', 'date', 'slot1', 'slot2', 'answer', 'ordered']).values
valid_Y = np.zeros((len(valid_X), 5))
for i, v in enumerate(valid_df['answer'].values):
	valid_Y[i, v] = 1
valid_Z = valid_df['ordered'].values.astype(int)

valid_count_db = {}
for ifp in set(valid_df['ifp_id']):
	index = valid_df.index[valid_df['ifp_id'] == ifp]
	valid_count_db[ifp] = 1.0/len(index)

valid_W = []
for index, row in valid_df.iterrows():
	valid_W.append(valid_count_db[row['ifp_id']])

valid_W = np.asarray(valid_W)
valid_W /= len(set(valid_df['ifp_id']))
valid_W *= len(valid_df)

test_X = test_df.drop(columns=['ifp_id', 'date', 'slot1', 'slot2', 'answer', 'ordered']).values
test_Y = np.zeros((len(test_X), 5))
for i, v in enumerate(test_df['answer'].values):
	test_Y[i, v] = 1
test_Z = test_df['ordered'].values

test_count_db = {}
for ifp in set(test_df['ifp_id']):
	index = test_df.index[test_df['ifp_id'] == ifp]
	test_count_db[ifp] = 1.0/len(index)

test_W = []
for index, row in test_df.iterrows():
	test_W.append(test_count_db[row['ifp_id']])

test_W = np.asarray(test_W)
test_W /= len(set(test_df['ifp_id']))
test_W *= len(test_df)

#pdb.set_trace()
X_placeholder = tf.placeholder(tf.float32, (None, 10), name='X')
Y_placeholder = tf.placeholder(tf.float32, (None, 5), name='Y')
Z_placeholder = tf.placeholder(tf.bool, (None, ), name='Z')
W_placeholder = tf.placeholder(tf.float32, (None, ), name='W')

W1 = tf.get_variable('W1', shape=(10, 20), initializer=tf.glorot_uniform_initializer())
b1 = tf.get_variable('b1', shape=(1, 20), initializer=tf.zeros_initializer())
W2 = tf.get_variable('W2', shape=(20, 15), initializer=tf.glorot_uniform_initializer())
b2 = tf.get_variable('b2', shape=(1, 15), initializer=tf.zeros_initializer())
W3 = tf.get_variable('W3', shape=(15, 5), initializer=tf.glorot_uniform_initializer())
b3 = tf.get_variable('b3', shape=(1, 5), initializer=tf.zeros_initializer())

l_output = tf.matmul(tf.nn.tanh(tf.matmul(tf.nn.tanh(tf.matmul(X_placeholder, W1) + b1), W2) + b2), W3) + b3
output = tf.nn.softmax(l_output)

lside_1 = tf.reduce_sum(tf.gather(output, [0], axis=1), axis=1)
rside_1 = tf.reduce_sum(tf.gather(output, [1, 2, 3, 4], axis=1), axis=1)
output_1 = tf.stack([lside_1, rside_1], axis=1)
lside_true_1 = tf.reduce_sum(tf.gather(Y_placeholder, [0], axis=1), axis=1)
rside_true_1 = tf.reduce_sum(tf.gather(Y_placeholder, [1, 2, 3, 4], axis=1), axis=1)
true_1 = tf.stack([lside_true_1, rside_true_1], axis=1)
loss_1 = tf.math.reduce_sum(tf.math.squared_difference(output_1, true_1), axis=1)

lside_2 = tf.reduce_sum(tf.gather(output, [0, 1], axis=1), axis=1)
rside_2 = tf.reduce_sum(tf.gather(output, [2, 3, 4], axis=1), axis=1)
output_2 = tf.stack([lside_2, rside_2], axis=1)
lside_true_2 = tf.reduce_sum(tf.gather(Y_placeholder, [0, 1], axis=1), axis=1)
rside_true_2 = tf.reduce_sum(tf.gather(Y_placeholder, [2, 3, 4], axis=1), axis=1)
true_2 = tf.stack([lside_true_2, rside_true_2], axis=1)
loss_2 = tf.math.reduce_sum(tf.math.squared_difference(output_2, true_2), axis=1)

lside_3 = tf.reduce_sum(tf.gather(output, [0, 1, 2], axis=1), axis=1)
rside_3 = tf.reduce_sum(tf.gather(output, [3, 4], axis=1), axis=1)
output_3 = tf.stack([lside_3, rside_3], axis=1)
lside_true_3 = tf.reduce_sum(tf.gather(Y_placeholder, [0, 1, 2], axis=1), axis=1)
rside_true_3 = tf.reduce_sum(tf.gather(Y_placeholder, [3, 4], axis=1), axis=1)
true_3 = tf.stack([lside_true_3, rside_true_3], axis=1)
loss_3 = tf.math.reduce_sum(tf.math.squared_difference(output_3, true_3), axis=1)

lside_4 = tf.reduce_sum(tf.gather(output, [0, 1, 2, 3], axis=1), axis=1)
rside_4 = tf.reduce_sum(tf.gather(output, [4], axis=1), axis=1)
output_4 = tf.stack([lside_4, rside_4], axis=1)

lside_true_4 = tf.reduce_sum(tf.gather(Y_placeholder, [0, 1, 2, 3], axis=1), axis=1)
rside_true_4 = tf.reduce_sum(tf.gather(Y_placeholder, [4], axis=1), axis=1)
true_4 = tf.stack([lside_true_4, rside_true_4], axis=1)
loss_4 = tf.math.reduce_sum(tf.math.squared_difference(output_4, true_4), axis=1)

loss_brier = tf.math.reduce_mean(tf.stack([loss_1, loss_2, loss_3, loss_4], axis=1), axis=1)
#loss_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y_placeholder, logits=l_output)
loss_mse = tf.math.reduce_sum(tf.math.squared_difference(Y_placeholder, output), axis=1)
#loss_mse = tf.losses.mean_squared_error(Y_placeholder, output)
#pdb.set_trace()
loss_combined = tf.where(Z_placeholder, loss_brier, loss_mse)
#pdb.set_trace()
loss_combined_reduce = tf.losses.compute_weighted_loss(loss_combined, W_placeholder)
#loss_combined_reduce = tf.math.reduce_mean(loss_combined)

loss_combined_reduce_reg = loss_combined_reduce
variables = [v for v in tf.trainable_variables() if 'W' in v.name]

for v in variables:
    loss_combined_reduce_reg += 0.0001 * tf.nn.l2_loss(v) + 0.001 * tf.losses.absolute_difference(v, tf.zeros(tf.shape(v)))


train_op = tf.train.AdamOptimizer(0.005).minimize(loss=loss_combined_reduce_reg)
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for epoch in range(25):
		train_step, loss_combined_reduce_train = sess.run(
		[train_op, loss_combined_reduce],
		feed_dict={
			X_placeholder: train_X,
			Y_placeholder: train_Y,
			Z_placeholder: train_Z,
			W_placeholder: train_W
		})

		[valid_Y_prob, loss_combined_reduce_valid] = sess.run(
		[output, loss_combined_reduce],
		feed_dict={
			X_placeholder: valid_X,
			Y_placeholder: valid_Y,
			Z_placeholder: valid_Z,
			W_placeholder: valid_W
		})

		#pdb.set_trace()
		[test_Y_prob, loss_combined_reduce_test] = sess.run(
		[output, loss_combined_reduce],
		feed_dict={
			X_placeholder: test_X,
			Y_placeholder: test_Y,
			Z_placeholder: test_Z,
			W_placeholder: test_W
		})

		print(epoch, loss_combined_reduce_train, loss_combined_reduce_valid, loss_combined_reduce_test)


		test_Y_brier = []
		for i, v in enumerate(test_Y_prob):
			ifp_id = test_df.iloc[i]['ifp_id']
			answer = test_df.iloc[i]['answer']
			ordered = test_df.iloc[i]['ordered']

			new_brier = brier_compute(v, answer, ordered)
			test_Y_brier.append(new_brier)

		test_Y_brier = np.asarray(test_Y_brier)

		db_brier = {}

		for ifp in set(test_df['ifp_id']):
			index = test_df.index[test_df['ifp_id'] == ifp]
			slot1 = test_df.iloc[index]['slot1'].values
			slot2 = test_df.iloc[index]['slot2'].values
			test_ideal = np.amin([slot1, slot2], axis=0)

			#pdb.set_trace()
			test_predict = test_Y_brier[index]
			db_brier[ifp] = [np.mean(slot1), np.mean(slot2), np.mean(test_ideal), np.mean(test_predict)]

		briers = np.asarray(list(db_brier.values()))
		briers_mean = np.mean(briers, axis=0)
		print(epoch, 'slot1', 'slot2', 'ideal', 'predict', 'test_loss')
		print(briers_mean, loss_combined_reduce_test)

	with open('briers.csv', 'w') as fout:
		fout.write('ifp,slot1,slot2,ideal,predict\n')
		for ifp, row in db_brier.items():
			fout.write(str(ifp) + ',' + ','.join([str(x) for x in row]) + '\n')
