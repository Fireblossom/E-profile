from models.lstm import load_vector
from models.lstm import LSTMModel
from models.lstm import res_prep

print("Start")
#wordemb, vecmat = load_vector('glove.6B.300d.txt')
wordemb, vecmat = load_vector('word_embedding_300_new.txt')
CONF = dict(veclen=300,
            maxlen=64,
            word_index=wordemb,
            embedding_matrix=vecmat
            )
print(len(wordemb), len(vecmat))
model = LSTMModel()
model.build_model(CONF)
model.model.summary()


from models.tools.Corpus import Corpus

#train_corpus = Corpus('train.csv')

#model.train(train_corpus, CONF['maxlen'], wordemb, 100, 64)
val_corpus = Corpus('val.csv')

result = model.predict(val_corpus, CONF['maxlen'], wordemb, 'train')
print(res_prep(result))
val_corpus.set_predict(res_prep(result))
val_corpus.set_label_title(('Anger', 'Anticipation', 'Disgust', 'Fear', 'Joy', 'Sadness', 'Surprise', 'Trust'))
print(val_corpus.eval())

print("OK")
