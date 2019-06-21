from linear import L_classifier
from corpus import Corpus

train_corpus = Corpus('train.csv')
model = L_classifier(8)
print('Begin training.')
model.train(train_corpus)

val_corpus = Corpus('val.csv')
val_corpus.set_predict(model.predict(val_corpus, train_corpus.tf_idf))

val_corpus.set_label_title(('Anger', 'Anticipation', 'Disgust', 'Fear', 'Joy', 'Sadness', 'Surprise', 'Trust'))
eval_result = val_corpus.eval()
print(eval_result)
