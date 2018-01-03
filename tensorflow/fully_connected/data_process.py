import pandas as pd

# -*- coding: UTF-8 -*-

# with open(r"../data/80w.txt", 'r', encoding='UTF-8') as f:
#     data = f.readlines()
#
# with open('data.txt', 'a',encoding='UTF-8') as file_object:
#     for i in range(len(data)):
#         file_object.write(data[i])

# data = pd.read_csv(r"../data/80w.txt", encoding='utf-8', sep='    ', header=None)
with open(r"../data/80w.txt", 'r', encoding='UTF-8') as f:
    data = f.readlines()

def data_process(data):
    label = []
    feature = []
    for i, item in enumerate(data):
        item = list(item)
        if i == 0:
            label.append(item[3])
            feature.append((''.join(item[5:])))
        else:
            if i < 9:
                label.append(item[2])
                feature.append(''.join(item[4:]))
            else:
                if i < 99:
                    label.append(item[3])
                    feature.append(''.join(item[5:]))
                else:
                    if i < 999:
                        label.append(item[4])
                        feature.append(''.join(item[6:]))
                    else:
                        if i< 9999:
                            label.append(item[5])
                            feature.append(''.join(item[7:]))
                        else:
                            if i < 99999:
                                label.append(item[6])
                                feature.append(''.join(item[8:]))
                            else:
                                if i < 999999:
                                    label.append(item[7])
                                    feature.append(''.join(item[9:]))
    return label, feature
print('生成原始数据。。。')
label, feature = data_process(data)
import jieba
import jieba.analyse








# print('开始生成停止词列表。。。')
# stopWords = []
# with open('../data/stopWord.txt', 'r', encoding='UTF-8') as f:
#
#     while 1:
#         line = f.readline().strip()
#         if not line:
#             break
#         stopWords.append(line)

jieba.analyse.set_stop_words('../data/stopWord.txt')
print('开始进行分词。。。。')
data = []
j = 0
for i in feature:
    j+=1
    if j % 10000 == 0:
        print(j)
    tags = jieba.analyse.extract_tags(i, 50)
    data.append(' '.join(list(tags)))


with open('data.txt','w') as fW:
    for i in range(len(data)):
        fW.write(data[i])
        fW.write('\n')

# import pickle
# with open('data.pkl', 'wb') as f :
#     pickle.dump(data, f)

3;








# s=jieba.cut(feature[1], cut_all=False)
# final = []
# for item in set(s):
#     if item is not in
# print(u'/'.join(set(s)))


# jieba.analyse.set_stop_words('../data/stopWord.txt')
# tags = jieba.analyse.extract_tags(feature[1],50)
# print(u'/'.join(set(tags)))
#
# with open('label.txt', 'w') as f:
#     f.writelines(label)
#
# with open('feature.txt', 'w',encoding='utf-8') as f:
#     f.writelines(feature)

