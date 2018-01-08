import pickle
# -*-coding:utf-8-*-

with open('../RNN/data.pkl', 'rb') as f:
    data_number = pickle.load(f)
with open('../RNN/label.pkl', 'rb') as f:
    labels = pickle.load(f)




seq_len = 50
import numpy as np

features = np.zeros((len(data_number), seq_len), dtype=int)
for i, row in enumerate(data_number):
    features[i, -len(row):] = np.array(row)[:seq_len]

from sklearn.model_selection import train_test_split

feature_train, feature_test, label_train, label_test = train_test_split(features, labels, test_size=0.3, random_state=0)

feature_test, feature_val, label_test, label_val = train_test_split(feature_test, label_test, test_size=0.3,random_state=0)

def get_batches(x, y, batch_size=100):
    n_batches = len(x) // batch_size
    x, y = x[:n_batches * batch_size], y[:n_batches * batch_size]
    for ii in range(0, len(x), batch_size):
        yield x[ii:ii + batch_size], y[ii:ii + batch_size]

print(u'读取原始数据集，构造训练以及测试数据集已经完成！')

print(u'*************************************************************')
batch_size = 1
import tensorflow as tf

with tf.variable_scope('input'):

    inputs = tf.placeholder(tf.int32, [batch_size, seq_len], name='inputs')
    labels = tf.placeholder(tf.float32, [batch_size, 1], name='labels')



keep_prob = 1

embed_size = 300
LAYER1_NODE = 512
LAYER2_NODE = 256
learning_rate = 0.01
with tf.variable_scope('network'):

    with tf.variable_scope('embeding'):
        embedding = tf.Variable(tf.random_uniform((8758, embed_size), -1, 1))
        embed = tf.nn.embedding_lookup(embedding, inputs)

    with tf.variable_scope('layer-1'):
        layer1 = tf.layers.dense(embed, LAYER1_NODE, activation=None)
        layer1 = tf.nn.dropout(layer1, keep_prob=keep_prob)
        # layer1 = tf.layers.batch_normalization(layer1, training=is_train)
        layer1 = tf.nn.relu(layer1)

    with tf.variable_scope('layer-2'):
        layer2 = tf.layers.dense(layer1, LAYER2_NODE, activation=None)
        layer2 = tf.nn.dropout(layer2, keep_prob=keep_prob)
        # layer2 = tf.layers.batch_normalization(layer2, training=is_train)
        layer2 = tf.nn.relu(layer2)

    with tf.variable_scope('output'):
        output = tf.layers.dense(layer2, 1, activation=None)
        output = output[:, -1]
with tf.variable_scope('train_op'):
    cost = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=output))
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)  # construct optimizer

with tf.variable_scope('predict_op'):
    predict_op = tf.nn.sigmoid(output)
    correct_pred = tf.equal(tf.cast(tf.round(predict_op), tf.float32), labels)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


saver = tf.train.Saver()
epochs = 1
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     iteration = 1
#     for e in range(epochs):
#         for ii, (x, y) in enumerate(get_batches(feature_train, label_train, batch_size), 1):
#             feed = {inputs: x,
#                     labels: y[:,None]}
#             loss, _,= sess.run([cost, train_op], feed_dict=feed)
#             if iteration % 5 == 0:
#                 print("Epoch: {}/{}".format(e, epochs),
#                     "Iteration: {}".format(iteration),
#                     "Train loss: {:.3f}".format(loss))
#             if iteration % 25 == 0:
#                 val_acc = []
#                 for x, y in get_batches(feature_val, label_val, batch_size):
#                     feed = {inputs: x,
#                             labels: y[:,None]}
#                     batch_acc, = sess.run([accuracy], feed_dict=feed)
#                     val_acc.append(batch_acc)
#
#                 print("Val acc: {:.3f}".format(np.mean(val_acc)))
#             iteration += 1
#             # saver.save(sess, os.path.join(os.getcwd(), "checkpoints/sentiment.ckpt"))
#             saver.save(sess, "./checkpoints/sentiment.ckpt")

def test1():
    inputWords = u'五一旅游的宝贝购物中大奖;低价降到底，好运转不停，黄金周不产黄金，五一节却有五折'
    inputWordsList = []
    import pickle
    with open('vocab_to_int.pkl', 'rb') as f:
        vocab_to_int = pickle.load( f)
    for item in inputWords:
        if item not in vocab_to_int.keys():
            inputWordsList.append(0)
        else:
            inputWordsList.append(vocab_to_int[item])
    seq_len = 50
    feature = np.zeros((1,seq_len), dtype=int)
    row = inputWordsList
    feature[0,-len(row):] = np.array(row)[:seq_len]
    label = np.ones((1,1))
    # feature[0, -len(row):] = np.array(row)[:seq_len]

    return feature, label
#
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, tf.train.latest_checkpoint('./checkpoints/'))
    # test_state = sess.run(cell.zero_state(batch_size, tf.float32))

    feed = {inputs: test1()[0],
            labels: test1()[1],
            }


    predictions = sess.run([predict_op], feed_dict=feed)

    print(((predictions)))









