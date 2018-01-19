import tensorflow as tf
import numpy as np
import pickle
with open('data.pkl', 'rb') as f:
    vocab_to_int, int_to_vocab, poems = pickle.load(f)
def get_tensors(loaded_graph):
    """
    Get input, initial state, final state, and probabilities tensor from <loaded_graph>
    :param loaded_graph: TensorFlow graph loaded from file
    :return: Tuple (InputTensor, InitialStateTensor, FinalStateTensor, ProbsTensor)
    """
    # TODO: Implement Function
    return loaded_graph.get_tensor_by_name("input:0"), loaded_graph.get_tensor_by_name("initial_state:0"), loaded_graph.get_tensor_by_name("final_state:0"), loaded_graph.get_tensor_by_name("probs:0")


def pick_word(probabilities, int_to_vocab):
    """
    Pick the next word in the generated text
    :param probabilities: Probabilites of the next word
    :param int_to_vocab: Dictionary of word ids as the keys and words as the values
    :return: String of the predicted word
    """
    # TODO: Implement Function
    # predicted_word = int_to_vocab[np.argmax(probabilities)]
    t = np.cumsum(probabilities)
    s = np.sum(probabilities)
    sample = int(np.searchsorted(t, np.random.rand(1) * s))
    if sample > len(int_to_vocab):
        sample = len(int_to_vocab) - 1
    return int_to_vocab[sample]
    # return predicted_word


prime_word = ''
gen_length = 128
seq_length = 128
load_dir = './checkpoint/save'


loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:

    # Load saved model
    loader = tf.train.import_meta_graph(load_dir + '.meta')
    loader.restore(sess, load_dir)
    # Get Tensors from loaded model
    input, initial_state, final_state, probs = get_tensors(loaded_graph)

    # Sentences generation setup
    # gen_sentences = [prime_word]
    prev_state = sess.run(initial_state, {input: np.array([[1]])})

    gen_sentences = []
    gen_sentences = ['S']
    for word in prime_word:
        gen_sentences.append(word)

    for n in range(gen_length):
        # Dynamic Input
        dyn_input = [[vocab_to_int[word] for word in gen_sentences[-seq_length:]]]
        dyn_seq_length = len(dyn_input[0])

        # Get Prediction
        probabilities, prev_state = sess.run(
            [probs, final_state],
            {input: dyn_input, initial_state: prev_state})

        pred_word = pick_word(probabilities[dyn_seq_length - 1], int_to_vocab)

        gen_sentences.append(pred_word)

        if pred_word == 'E':
            break



        # Remove tokens

def printItem(data):
    poems = ''
    for word in data:
        if word =='S':
            continue
        else:
            if word =='C':
                poems += '，  '
                continue
            else:
                if word == 'P':
                    poems += '。 '+'\n'

                    continue
                else:
                    if word == 'E':
                        poems += '。 '
                        break
                    else:
                        poems += word
    print(poems)
printItem(gen_sentences)