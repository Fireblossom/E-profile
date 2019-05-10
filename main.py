from models.naive_bayes import NB_classifier
import models.tools.file_writer
from models.tools.Corpus import Corpus

train_corpus = Corpus('gold.csv')

model = NB_classifier(8)
model.train(train_corpus)

val_corpus = Corpus('gold.csv')
val_corpus.set_predict(model.predict(val_corpus))
val_corpus.set_label_title(('Anger', 'Anticipation', 'Disgust', 'Fear', 'Joy', 'Sadness', 'Surprise', 'Trust'))
eval_result = val_corpus.eval()
print(eval_result)

# pred_file = open('predict.csv', 'a')


