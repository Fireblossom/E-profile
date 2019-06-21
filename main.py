from linear import L_classifier
from corpus import Corpus
import find_false
from corpus import dict_generator

dicts = dict_generator('features')

train_corpus = Corpus('train.csv')
model = L_classifier(8)
print('Begin training.')
model.train(train_corpus, dicts)

val_corpus = Corpus('val.csv')
val_corpus.set_predict(model.predict(val_corpus, train_corpus.tf_idf, dicts))

val_corpus.set_label_title(('Anger', 'Anticipation', 'Disgust', 'Fear', 'Joy', 'Sadness', 'Surprise', 'Trust'))
eval_result = val_corpus.eval()
print(eval_result)

print(find_false.find_fp(val_corpus))
