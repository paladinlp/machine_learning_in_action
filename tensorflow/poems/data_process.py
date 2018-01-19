# -*- coding: UTF-8 -*-
# with open('../data/poems.txt', 'r', encoding='utf-8') as f:
#     data = f.readlines()
# from collections import Counter
# counts =Counter()
# vocab = []
# # # 5046 43030
# data_process = []
# start_token = 'S'
# for item in data:
#     item =item.strip()
#     item = item[item.index(':')+1:-1]
#     item = item.replace('，', 'C')
#     item = item.replace('。', 'P')
#     item = start_token + item + 'E'
#     data_process.append(item)
#     for word in list(item):
#         vocab.append(word)
# counts = Counter(vocab)
# vocab = sorted(counts, key=counts.get, reverse=True)
# vocab_to_int = {word:i for i, word in enumerate(vocab)}
# print(len(vocab_to_int))
# int_to_vocab = {i:word for i, word in enumerate(vocab)}
#
# poems = [vocab_to_int[word] for item in data_process for word in list(item) ]
# # # poems = []
# # for item in data_process:
# #
# #     for word in list(item):
# #         poems.append(vocab_to_int[word])
#
import pickle
# with open('data.pkl', 'wb') as f :
#     pickle.dump((vocab_to_int, int_to_vocab, poems), f)

with open('data.pkl', 'rb') as f:
    vocab_to_int, int_to_vocab, poems = pickle.load(f)
import numpy as np

vocab_size = 7561
embed_dim = 300
batch_size = 64
seq_length = 64
rnn_size = 256
num_epochs = 500
learning_rate = 0.001
show_every_n_batches = 1000
save_dir = './checkpoint/save'

def get_batches(int_text, batch_size, seq_length):
    batch = batch_size*seq_length
    int_text = int_text[:(len(int_text)//(batch))*(batch)]
    int_text = np.reshape(int_text, (batch_size, -1))
    batch_list = []
    for n in range(0, int_text.shape[1], seq_length):
        x = int_text[:, n:n+seq_length]
        y = np.zeros_like(x)
        if int_text.shape[1] > n+seq_length:
            y[:, :-1], y[:, -1] = x[:, 1:], int_text[:, n+seq_length]
        else:
            y[:, :-1], y[:-1, -1], y[-1, -1] = x[:, 1:], int_text[1:, 0], int_text[0, 0]
        one_batch = np.array([x, y])
        batch_list.append(one_batch)
    return np.array(batch_list)

import tensorflow as tf

train_graph = tf.Graph()
with train_graph.as_default():
    # with tf.variable_scope('inputs'):
    input = tf.placeholder(tf.int32, [None, None], name='input')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')
    lr = tf.placeholder(tf.float32)


    # with tf.variable_scope('embeding'):
    embedding = tf.Variable(tf.random_uniform((vocab_size, embed_dim), -1, 1))
    embed = tf.nn.embedding_lookup(embedding, input)

    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)

    # Add dropout to the cell
    drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=1.0)
    cell = tf.contrib.rnn.MultiRNNCell([drop] * 2)

    initial_state = tf.identity(cell.zero_state(tf.shape(input)[0], tf.float32), name='initial_state')
    outputs, final_state = tf.nn.dynamic_rnn(cell, embed, dtype=tf.float32)
    final_state = tf.identity(final_state, name='final_state')

    # with tf.variable_scope('fully_connected'):
    logits = tf.contrib.layers.fully_connected(outputs, vocab_size, activation_fn=None)
    probs = tf.nn.softmax(logits, name='probs')

    input_data_shape = tf.shape(input)


    from tensorflow.contrib import seq2seq
    cost = seq2seq.sequence_loss(
        logits,
        targets,
        tf.ones([input_data_shape[0], input_data_shape[1]]))

    optimizer = tf.train.AdamOptimizer(learning_rate)
    # train_op = optimizer.minimize(cost)
    # Gradient Clipping
    gradients = optimizer.compute_gradients(cost)
    capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
    train_op = optimizer.apply_gradients(capped_gradients)

batches = get_batches(poems, batch_size, seq_length)

with tf.Session(graph=train_graph) as sess:
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    for epoch_i in range(num_epochs):
        state = sess.run(initial_state, {input: batches[0][0]})

        for batch_i, (x, y) in enumerate(batches):
            feed = {
                input: x,
                targets: y,
                initial_state: state,
                lr: learning_rate}
            train_loss, state, _ = sess.run([cost, final_state, train_op], feed)

            # Show every <show_every_n_batches> batches
            if (epoch_i * len(batches) + batch_i) % show_every_n_batches == 0:
                print('Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}'.format(
                    epoch_i,
                    batch_i,
                    len(batches),
                    train_loss))


    # Save Model
                saver.save(sess, save_dir)
                print('Model Trained and Saved')


