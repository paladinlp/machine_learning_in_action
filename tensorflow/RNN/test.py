# -*-coding:utf-8-*-
__author__ = 'paladinlp'
__date__ = '2017/12/29 21:58'

import numpy as np

# with open('../fully_connected/80W.txt','r',encoding='utf-8') as f:
#     data = f.readlines()
#
#
# labels = np.ones((800000),dtype=int)
# for i, item in enumerate(data):
#     item = list(item)
#     if i == 0:
#         labels[i] = item[2]
#     else:
#         if i < 9:
#             labels[i] = item[2]
#         else:
#             if i < 99:
#                 labels[i] = item[3]
#             else:
#                 if i < 999:
#                     labels[i] = item[4]
#                 else:
#                     if i < 9999:
#                         labels[i] = item[5]
#                     else:
#                         if i < 99999:
#                             labels[i] = item[6]
#                         else:
#                             if i < 999999:
#
#                                 labels[i] = item[7]
#                             else:
#                                 if i < 9999999:
#                                     labels[i] = item[8]


import random





# def test():
#     is_error = False
#     for j in range(1000):
#         i = random.randint(0, 799999)
#         item = list(data[i])
#         if i == 0:
#             a = item[2]
#         else:
#             if i < 9:
#                 a= item[2]
#             else:
#                 if i < 99:
#                     a = item[3]
#                 else:
#                     if i < 999:
#                         a= item[4]
#                     else:
#                         if i < 9999:
#                             a= item[5]
#                         else:
#                             if i < 99999:
#                                 a = item[6]
#                             else:
#                                 if i < 999999:
#
#                                     a = item[7]
#                                 else:
#                                     if i < 9999999:
#                                         a= item[8]
#             a = int(a)
#             b = int(labels[i])
#             if a != b:
#                 print("error!")
#                 is_error = True
#                 break
#     if   is_error is False:
#         print("right")


import numpy as np
def test1():
    inputWords = u'房地产信息等等，请问您有购房的需求吗'
    inputWordsList = []
    import pickle
    with open('vocab_to_int.pkl', 'rb') as f:
        vocab_to_int = pickle.load( f)
    for item in inputWords:
        if item  not in vocab_to_int.keys():
            inputWordsList.append(0)
        else:
            inputWordsList.append(vocab_to_int[item])
    seq_len = 100
    feature = np.zeros((1,seq_len), dtype=int)
    row = inputWordsList
    feature[0,-len(row):] = np.array(row)[:seq_len]
    label = np.ones((1,int(1)))
    # feature[0, -len(row):] = np.array(row)[:seq_len]
    print(feature,label)
if __name__ == '__main__':
    pass


