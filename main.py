from models.naive_bayes import NB_classifier
from models.linear import L_classifier
from models.tools.Corpus import Corpus
from models.tools.Corpus import dict_generator
from models.lstm import syntax_features

print("Choose model: \n1.Perceptron, 2.Naive Bayes, 3.ON-LSTM, 4.LSTM")
select = int(input())

train_corpus = Corpus('train.csv')
val_corpus = Corpus('test.csv')
val_corpus.set_label_title(('Anger', 'Anticipation', 'Disgust', 'Fear', 'Joy', 'Sadness', 'Surprise', 'Trust'))

if select == 1:
    lexical = dict_generator('features')
    train_syntax = syntax_features(train_corpus)
    test_syntax = syntax_features(val_corpus)
    model = L_classifier(8)
    model.train(train_corpus, dicts=lexical, syntax_feature=train_syntax)
    val_corpus.set_predict(model.predict(val_corpus, train_corpus.tf_idf, dicts=lexical, syntax_feature=test_syntax))
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
