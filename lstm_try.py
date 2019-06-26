from models.lstm import load_vector
from models.lstm import LSTMModel
from models.tools.Corpus import Corpus, dict_generator
import numpy as np


def run():
    print("Start")
    wordemb, vecmat = load_vector('new_embdding.txt')
    embedding = merge_embdding(wordemb, vecmat)

    CONF = dict(veclen=len(embedding[0]),
                maxlen=64,
                word_index=wordemb,
                embedding_matrix=embedding
                # syntax_features=
                )
    model = LSTMModel()
    model.build_model(CONF)
    model.model.summary()

    #train_corpus = Corpus('train.csv')

    #model.train(train_corpus, CONF['maxlen'], wordemb, 100, 64)
    val_corpus = Corpus('val.csv')

    result = model.predict(val_corpus, CONF['maxlen'], wordemb, 'train')
    print(result)
    val_corpus.set_predict(result)
    val_corpus.set_label_title(('Anger', 'Anticipation', 'Disgust', 'Fear', 'Joy', 'Sadness', 'Surprise', 'Trust'))
    print(val_corpus.eval())

    print("OK")


def merge_embdding(wordemb, vecmat):
    dicts = dict_generator('features')
    features = np.zeros((vecmat.shape[0], len(dicts)))
    for word in wordemb:
        for i in range(len(dicts)):
            if word not in dicts[i]:
                features[wordemb[word]][i] = -1
            elif dicts[i][word] == 0:
                features[wordemb[word]][i] = -1
            elif dicts[i][word] > 0:
                features[wordemb[word]][i] = 1
    new_vecmat = np.column_stack([vecmat, features])
    return new_vecmat

