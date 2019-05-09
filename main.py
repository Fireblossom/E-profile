from models.naive_bayes import NB_classifier
import models.tools.file_writer
from models.tools.Corpus import Corpus

train_corpus = Corpus('train.csv')

model = NB_classifier(8)
model.train(train_corpus)

val_corpus = Corpus('val.csv')
val_corpus.set_predict(model.predict(val_corpus))
eval_result = val_corpus.eval()

pred_file = open('predict.csv', 'a')
label_title = ('Anger', 'Anticipation', 'Disgust', 'Fear', 'Joy', 'Sadness', 'Surprise', 'Trust')

