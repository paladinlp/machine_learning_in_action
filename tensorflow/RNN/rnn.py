# -*-coding:utf-8-*-
__author__ = 'paladinlp'
__date__ = '2017/12/30 9:11'

import os





import pickle
with open('data.pkl', 'rb') as f:
    data_number = pickle.load(f)
with open('label.pkl', 'rb') as f:
    label = pickle.load(f)
seq_len = 200
import numpy as np

feature = np.zeros((len(data_number),seq_len), dtype = int)
for i,row in enumerate(data_number):
    feature[i, -len(row):] = np.array(row)[:seq_len]



from sklearn.model_selection import train_test_split

feature_train,feature_test, label_train, label_test = train_test_split(feature, label, test_size=0.3, random_state=0)

feature_test,feature_val, label_test, label_val = train_test_split(feature_test, label_test, test_size=0.3, random_state=0)

import tensorflow as tf

graph = tf.Graph()
# Add nodes to the graph
with graph.as_default():
    inputs_ = tf.placeholder(tf.int32, [None, None], name='inputs')
    labels_ = tf.placeholder(tf.int32, [None, None], name='labels')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

embed_size = 300

with graph.as_default():
    embedding = tf.Variable(tf.random_uniform((8758, embed_size), -1, 1))
    embed = tf.nn.embedding_lookup(embedding, inputs_)


lstm_size = 256
lstm_layers = 1
batch_size = 1
learning_rate = 0.001

with graph.as_default():
    # Your basic LSTM cell
    lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
    # Add dropout to the cell
    drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
    # Stack up multiple LSTM layers, for deep learning
    cell = tf.contrib.rnn.MultiRNNCell([drop] * lstm_layers)
    # Getting an initial state of all zeros
    initial_state = cell.zero_state(batch_size, tf.float32)

with graph.as_default():
    outputs, final_state = tf.nn.dynamic_rnn(cell, embed,
                                             initial_state=initial_state)

with graph.as_default():

    predictions = tf.contrib.layers.fully_connected(outputs[:, -1], 1, activation_fn=tf.sigmoid)
    cost = tf.losses.mean_squared_error(labels_, predictions)

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

with graph.as_default():
    correct_pred = tf.equal(tf.cast(tf.round(predictions), tf.int32), labels_)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


def get_batches(x, y, batch_size=100):
    n_batches = len(x) // batch_size
    x, y = x[:n_batches * batch_size], y[:n_batches * batch_size]
    for ii in range(0, len(x), batch_size):
        yield x[ii:ii + batch_size], y[ii:ii + batch_size]


epochs = 1

with graph.as_default():
    saver = tf.train.Saver()

# with tf.Session(graph=graph) as sess:
#     sess.run(tf.global_variables_initializer())
#     iteration = 1
#     for e in range(epochs):
#         state = sess.run(initial_state)
#
#         for ii, (x, y) in enumerate(get_batches(feature_train, label_train, batch_size), 1):
#             feed = {inputs_: x,
#                     labels_: y[:, None],
#                     keep_prob: 0.5,
#                     initial_state: state}
#             loss, state, _ = sess.run([cost, final_state, optimizer], feed_dict=feed)
#
#             if iteration % 5 == 0:
#                 print("Epoch: {}/{}".format(e, epochs),
#                       "Iteration: {}".format(iteration),
#                       "Train loss: {:.3f}".format(loss))
#
#             if iteration % 25 == 0:
#                 val_acc = []
#                 val_state = sess.run(cell.zero_state(batch_size, tf.float32))
#                 for x, y in get_batches(feature_val, label_val, batch_size):
#                     feed = {inputs_: x,
#                             labels_: y[:, None],
#                             keep_prob: 1,
#                             initial_state: val_state}
#                     batch_acc, val_state = sess.run([accuracy, final_state], feed_dict=feed)
#                     val_acc.append(batch_acc)
#                 print("Val acc: {:.3f}".format(np.mean(val_acc)))
#                 break
#             iteration += 1
#
#     # saver.save(sess, os.path.join(os.getcwd(), "checkpoints/sentiment.ckpt"))
#     saver.save(sess, "./checkpoints/sentiment.ckpt")

# test_acc = []
# print('进行实际测试')
# with tf.Session(graph=graph) as sess:
#     saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
#     test_state = sess.run(cell.zero_state(batch_size, tf.float32))
#     for ii, (x, y) in enumerate(get_batches(feature_test, label_test, batch_size), 1):
#
#         feed = {inputs_: x,
#                 labels_: y[:, None],
#                 keep_prob: 1,
#                 initial_state: test_state}
#         batch_acc, test_state = sess.run([accuracy, final_state], feed_dict=feed)
#         test_acc.append(batch_acc)
#     print("Test accuracy: {:.3f}".format(np.mean(test_acc)))

# 进行判别测试

def test1():
    inputWords = u'五一旅游的宝贝，XX手机低价呈现，购物中大奖;低价降到底，好运转不停，黄金周不产黄金，五一节却有五折'
    inputWordsList = []
    import pickle
    with open('vocab_to_int.pkl', 'rb') as f:
        vocab_to_int = pickle.load( f)
    for item in inputWords:
        if item not in vocab_to_int.keys():
            inputWordsList.append(0)
        else:
            inputWordsList.append(vocab_to_int[item])
    seq_len = 200
    feature = np.zeros((1,seq_len), dtype=int)
    row = inputWordsList
    feature[0,-len(row):] = np.array(row)[:seq_len]
    label = np.ones((1,1))
    # feature[0, -len(row):] = np.array(row)[:seq_len]
    print(feature)
    return feature, label

with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, tf.train.latest_checkpoint('./checkpoints'))
    test_state = sess.run(cell.zero_state(batch_size, tf.float32))

    feed = {inputs_: test1()[0],
            labels_: test1()[1],
            keep_prob: 1,
            initial_state: test_state}
    predictions, test_state = sess.run([predictions, final_state], feed_dict=feed)
    print(sess.run(tf.round(predictions)))






if __name__ == '__main__':
    pass