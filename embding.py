from models.tools.Corpus import Corpus

train_corpus = Corpus('train.csv')
vocab = set()
for text in train_corpus.text:
    for word in text.words:
        vocab.add(word)

with open('glove.6B.300d.txt') as file:
    f = open('new_embdding.txt', 'w')
    for line in file:
        if line.split(' ')[0] in vocab:
            f.write(line)
f.close()
