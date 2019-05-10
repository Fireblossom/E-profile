from models.tools.Corpus import Corpus
import nltk


want_tags = {'JJ', 'JJR', 'JJS'}

train_corpus = Corpus('train.csv')
adj = []
for inst in train_corpus.text:
    pos_tags = nltk.pos_tag(inst.words)
    for word, pos in pos_tags:
        if pos in want_tags:
            adj.append(word)


from collections import Counter
word_counts = Counter(adj)
top_100 = word_counts.most_common(100)
print(top_100)
