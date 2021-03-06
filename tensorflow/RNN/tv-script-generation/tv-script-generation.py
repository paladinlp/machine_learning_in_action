# """
# DON'T MODIFY ANYTHING IN THIS CELL
# """
# import helper
#
# data_dir = './data/simpsons/moes_tavern_lines.txt'
# text = helper.load_data(data_dir)
# # Ignore notice, since we don't use it for analysing the data
# text = text[81:]
# print(text.split())
#
# # view_sentence_range = (0, 10)
# #
# # """
# # DON'T MODIFY ANYTHING IN THIS CELL
# # """
# # import numpy as np
# #
# # print('Dataset Stats')
# # print('Roughly the number of unique words: {}'.format(len({word: None for word in text.split()})))
# # scenes = text.split('\n\n')
# # print('Number of scenes: {}'.format(len(scenes)))
# # sentence_count_scene = [scene.count('\n') for scene in scenes]
# # print('Average number of sentences in each scene: {}'.format(np.average(sentence_count_scene)))
# #
# # sentences = [sentence for scene in scenes for sentence in scene.split('\n')]
# # print('Number of lines: {}'.format(len(sentences)))
# # word_count_sentence = [len(sentence.split()) for sentence in sentences]
# # print('Average number of words in each line: {}'.format(np.average(word_count_sentence)))
# #
# # print()
# # print('The sentences {} to {}:'.format(*view_sentence_range))
# # print('\n'.join(text.split('\n')[view_sentence_range[0]:view_sentence_range[1]]))
#
#
# import numpy as np
# import problem_unittests as tests
#
# def create_lookup_tables(text):
#     """
#     Create lookup tables for vocabulary
#     :param text: The text of tv scripts split into words
#     :return: A tuple of dicts (vocab_to_int, int_to_vocab)
#     """
#     # TODO: Implement Function
#     vocab = sorted(set(text))
#     vocab_to_int = {word: i for i, word in enumerate(vocab)}
#     int_to_vocab = {i:word for i,word in enumerate(vocab)}
#     return vocab_to_int, int_to_vocab
#
#
# """
# DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
# """
# tests.test_create_lookup_tables(create_lookup_tables)
#
#
# def token_lookup():
#     """
#     Generate a dict to turn punctuation into a token.
#     :return: Tokenize dictionary where the key is the punctuation and the value is the token
#     """
#     # TODO: Implement Function
#     return {
#         '.': '||Period||',
#         ',': '||Comma||',
#         "\"": '||QuotationMark||',
#         ';': '||Semicolon||',
#         '!': '||ExclamationMark||',
#         '?': '||QuestionMark||',
#         '(': '||LeftParentheses||',
#         ')': '||RightParentheses||',
#         '--': '||Dash||',
#         '\n': '||Return||', }
#
# """
# DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
# """
# tests.test_tokenize(token_lookup)
#
#
# helper.preprocess_and_save_data(data_dir, token_lookup, create_lookup_tables)


# -*-coding:utf-8-*-


import helper
import numpy as np
import problem_unittests as tests

int_text, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
# from distutils.version import LooseVersion
# import warnings
import tensorflow as tf
#
# # Check TensorFlow Version
# assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer'
# print('TensorFlow Version: {}'.format(tf.__version__))
#
# # Check for a GPU
# if not tf.test.gpu_device_name():
#     warnings.warn('No GPU found. Please use a GPU to train your neural network.')
# else:
#     print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

def get_inputs():
    """
    Create TF Placeholders for input, targets, and learning rate.
    :return: Tuple (input, targets, learning rate)
    """
    # TODO: Implement Function

    inputs = tf.placeholder(tf.int32, [None, None],name = 'input')

    targets = tf.placeholder(tf.int32, [None,None],name = 'targets')

    learning_rate = tf.placeholder(tf.float32, name='learning_rate')

    return inputs, targets, learning_rate


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
# tests.test_get_inputs(get_inputs)

def get_init_cell(batch_size, rnn_size):
    """
    Create an RNN Cell and initialize it.
    :param batch_size: Size of batches
    :param rnn_size: Size of RNNs
    :return: Tuple (cell, initialize state)
    """
    # TODO: Implement Function
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    # Stack up multiple LSTM layers, for deep learning
    cell = tf.contrib.rnn.MultiRNNCell([lstm] * 2)
    initial_state = tf.identity(
        cell.zero_state(batch_size, tf.float32),name='initial_state'
)
    return cell, initial_state


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
# tests.test_get_init_cell(get_init_cell)

def get_embed(input_data, vocab_size, embed_dim):
    """
    Create embedding for <input_data>.
    :param input_data: TF placeholder for text input.
    :param vocab_size: Number of words in vocabulary.
    :param embed_dim: Number of embedding dimensions
    :return: Embedded input.
    """
    # TODO: Implement Function
    with tf.variable_scope('embedding'):
        embedding = tf.Variable(tf.random_uniform((vocab_size, embed_dim), -1, 1))
        embed = tf.nn.embedding_lookup(embedding, input_data)
    return embed


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
# tests.test_get_embed(get_embed)

def build_rnn(cell, inputs):
    """
    Create a RNN using a RNN Cell
    :param cell: RNN Cell
    :param inputs: Input text data
    :return: Tuple (Outputs, Final State)
    """
    # TODO: Implement Function
    outputs, final_state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32
                                             )


    final_state =  tf.identity(
        final_state,name='final_state',
)

    return outputs, final_state


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
def build_nn(cell, rnn_size, input_data, vocab_size, embed_dim):
    """
    Build part of the neural network
    :param cell: RNN cell
    :param rnn_size: Size of rnns
    :param input_data: Input data
    :param vocab_size: Vocabulary size
    :param embed_dim: Number of embedding dimensions
    :return: Tuple (Logits, FinalState)
    """
    # TODO: Implement Function
    embed =  get_embed(input_data, vocab_size, embed_dim)

    outputs, final_state = build_rnn(cell, embed)

    # # outputs=tf.concat(outputs,axis = 1)
    #
    # outputs = tf.reshape(outputs,[-1, rnn_size])

    outputs = tf.contrib.layers.fully_connected(outputs, vocab_size, activation_fn=None)


    return outputs, final_state


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""


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
tests.test_get_batches(get_batches)

# Number of Epochs
num_epochs = 200
# Batch Size
batch_size = 128
# RNN Size
rnn_size = 256
# Embedding Dimension Size
embed_dim = 300
# Sequence Length
seq_length = 15
# Learning Rate
learning_rate = 0.001
# Show stats for every n number of batches
show_every_n_batches = 200

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
save_dir = './save'

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
from tensorflow.contrib import seq2seq

train_graph = tf.Graph()
with train_graph.as_default():
    vocab_size = len(int_to_vocab)
    input_text, targets, lr = get_inputs()
    input_data_shape = tf.shape(input_text)
    cell, initial_state = get_init_cell(input_data_shape[0], rnn_size)
    logits, final_state = build_nn(cell, rnn_size, input_text, vocab_size, embed_dim)

    # Probabilities for generating words
    probs = tf.nn.softmax(logits, name='probs')

    # Loss function
    cost = seq2seq.sequence_loss(
        logits,
        targets,
        tf.ones([input_data_shape[0], input_data_shape[1]]))

    # Optimizer
    optimizer = tf.train.AdamOptimizer(lr)

    # Gradient Clipping
    gradients = optimizer.compute_gradients(cost)
    capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
    train_op = optimizer.apply_gradients(capped_gradients)

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
batches = get_batches(int_text, batch_size, seq_length)

with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch_i in range(num_epochs):
        state = sess.run(initial_state, {input_text: batches[0][0]})

        for batch_i, (x, y) in enumerate(batches):
            feed = {
                input_text: x,
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
    saver = tf.train.Saver()
    saver.save(sess, save_dir)
    print('Model Trained and Saved')

