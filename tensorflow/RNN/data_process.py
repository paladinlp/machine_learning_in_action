with open('feature.txt','r',encoding='UTF-8') as f:
    data = f.readlines()

vocab = []
for item in data:
    for word in item:

        vocab.append(word)

from collections import Counter

counts =Counter(vocab)

vocab = sorted(counts, key = counts.get, reverse=True)



vocab_to_int = { c: i for i, c in enumerate(vocab)}

print(len(vocab_to_int))


import pickle
with open('vocab_to_int.pkl', 'wb') as f:
    pickle.dump(vocab_to_int, f)


data_number = []
for item in data:
    word = []
    for i in item[:-1]:
        word.append(vocab_to_int[i])
    data_number.append(word)
# import pickle
# with open('data.pkl', 'wb') as f:
#     pickle.dump(data_number, f)

# with open('data.pkl', 'rb') as f:
#     data_number = pickle.load(f)
#
# seq_len = 100
#
#
# import numpy as np
# feature = np.zeros((len(data_number),seq_len),dtype = int)
# for i,row in enumerate(data_number):
#     feature[i, -len(row):] = np.array(row)[:seq_len]
#
# print(feature[100])


