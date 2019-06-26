from linear import L_classifier
from corpus import Corpus
import find_false
from corpus import dict_generator
from collections import Counter

dicts = dict_generator('features')

train_corpus = Corpus('train.csv')
model = L_classifier(8)
print('Begin training.')
model.train(train_corpus, dicts)
print(model.w[0])

val_corpus = Corpus('val.csv')
val_corpus.set_predict(model.predict(val_corpus, train_corpus.tf_idf, dicts))

val_corpus.set_label_title(('Anger', 'Anticipation', 'Disgust', 'Fear', 'Joy', 'Sadness', 'Surprise', 'Trust'))
eval_result = val_corpus.eval()
print(eval_result)

word_list_n = Counter(find_false.find_fn(val_corpus, 6))
word_list_p = Counter(find_false.find_fp(val_corpus, 6))

with open('Surprise_fn.txt', 'w') as file:
    for word in word_list_n:
        file.write(word + '\t' + str(word_list_n[word] + '\n'))

with open('Surprise_fp.txt', 'w') as file:
    for word in word_list_p:
        file.write(word + '\t' + str(word_list_p[word] + '\n'))
