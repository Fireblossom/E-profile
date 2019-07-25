from models.naive_bayes import NB_classifier
from models.linear import L_classifier
from models.tools.Corpus import Corpus

print("Choose model:\n1.Perceptron, 2.Naive Bayes, 3.LSTM, 4.ON-LSTM")
select = int(input())

train_corpus = Corpus('train.csv')
val_corpus = Corpus('val.csv')
val_corpus.set_label_title(('Anger', 'Anticipation', 'Disgust', 'Fear', 'Joy', 'Sadness', 'Surprise', 'Trust'))
if select == 1:
    model = L_classifier(8)
    model.train(train_corpus)
    val_corpus.set_predict(model.predict(val_corpus, train_corpus.tf_idf))
    eval_result = val_corpus.eval()
    print(eval_result)

elif select == 2:
    model = NB_classifier(8)
    model.train(train_corpus)
    val_corpus.set_predict(model.predict(val_corpus))
    eval_result = val_corpus.eval()
    print(eval_result)

elif select == 3:
    import auto_on

elif select == 4:
    import auto_lstm
