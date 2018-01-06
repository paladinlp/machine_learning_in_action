"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import helper

data_dir = './data/simpsons/moes_tavern_lines.txt'
text = helper.load_data(data_dir)
# Ignore notice, since we don't use it for analysing the data
text = text[81:]
print(text.split())

# view_sentence_range = (0, 10)
#
# """
# DON'T MODIFY ANYTHING IN THIS CELL
# """
# import numpy as np
#
# print('Dataset Stats')
# print('Roughly the number of unique words: {}'.format(len({word: None for word in text.split()})))
# scenes = text.split('\n\n')
# print('Number of scenes: {}'.format(len(scenes)))
# sentence_count_scene = [scene.count('\n') for scene in scenes]
# print('Average number of sentences in each scene: {}'.format(np.average(sentence_count_scene)))
#
# sentences = [sentence for scene in scenes for sentence in scene.split('\n')]
# print('Number of lines: {}'.format(len(sentences)))
# word_count_sentence = [len(sentence.split()) for sentence in sentences]
# print('Average number of words in each line: {}'.format(np.average(word_count_sentence)))
#
# print()
# print('The sentences {} to {}:'.format(*view_sentence_range))
# print('\n'.join(text.split('\n')[view_sentence_range[0]:view_sentence_range[1]]))


import numpy as np
import problem_unittests as tests

def create_lookup_tables(text):
    """
    Create lookup tables for vocabulary
    :param text: The text of tv scripts split into words
    :return: A tuple of dicts (vocab_to_int, int_to_vocab)
    """
    # TODO: Implement Function
    vocab = sorted(set(text))
    vocab_to_int = {word: i for i, word in enumerate(vocab)}
    int_to_vocab = {i:word for i,word in enumerate(vocab)}
    return vocab_to_int, int_to_vocab


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_create_lookup_tables(create_lookup_tables)


def token_lookup():
    """
    Generate a dict to turn punctuation into a token.
    :return: Tokenize dictionary where the key is the punctuation and the value is the token
    """
    # TODO: Implement Function
    return {
        '.': '||Period||',
        ',': '||Comma||',
        "\"": '||QuotationMark||',
        ';': '||Semicolon||',
        '!': '||ExclamationMark||',
        '?': '||QuestionMark||',
        '(': '||LeftParentheses||',
        ')': '||RightParentheses||',
        '--': '||Dash||',
        '\n': '||Return||', }

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_tokenize(token_lookup)


helper.preprocess_and_save_data(data_dir, token_lookup, create_lookup_tables)