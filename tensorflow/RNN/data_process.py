with open('feature.txt','r',encoding='UTF-8') as f:
    data = f.readlines()

vocab = set([])
for item in data:
    for word in item:
        vocab.add(word)



vocab_to_int = { c: i for i, c in enumerate(vocab)}
data_number = []
for item in data:
    word = []
    for i in item[:-1]:
        word.append(vocab_to_int[i])
    data_number.append(word)

