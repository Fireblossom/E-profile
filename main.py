from models.naive_bayes import NB_classifier
from models.linear import L_classifier
from models.tools.Corpus import Corpus

train_corpus = Corpus('train.csv')
#model = L_classifier(8)
#model.train(train_corpus)
# print(model.w[0])
print(train_corpus.pos)
# sentence = "I feel the greatest destroyer of peace today is 'Abortion',... Mother Teresa #SemST"
# print(model.test_sentence(sentence, train_corpus.tf_idf))

#val_corpus = Corpus('val.csv')
# val_corpus.set_predict(model.predict(val_corpus, train_corpus.tf_idf))
#val_corpus.set_predict(model.predict(val_corpus, train_corpus.tf_idf))

#val_corpus.set_label_title(('Anger', 'Anticipation', 'Disgust', 'Fear', 'Joy', 'Sadness', 'Surprise', 'Trust'))
#eval_result = val_corpus.eval()
#print(eval_result)

# pred_file = open('predict.csv', 'a')


