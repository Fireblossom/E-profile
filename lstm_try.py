from models.lstm import LSTMModel, load_vector, merge_embdding, syntax_features
from models.tools.Corpus import Corpus


def run():
    print("Start")
    wordemb, vecmat = load_vector('new_embdding.txt')
    embedding = merge_embdding(wordemb, vecmat)
    train_corpus = Corpus('train.csv')
    manual_features = syntax_features(train_corpus)
    print(manual_features)

    CONF = dict(veclen=len(embedding[0]),
                maxlen=64,
                word_index=wordemb,
                embedding_matrix=embedding,
                syntax_features=manual_features
                )
    model = LSTMModel()

    model.build_model(CONF)
    model.model.summary()

    model.train(train_corpus, CONF, 100, 64)

    val_corpus = Corpus('test.csv')
    CONF['syntax_features'] = syntax_features(val_corpus)

    result = model.predict(val_corpus, CONF, 'train')
    print(result)
    val_corpus.set_predict(result)
    title = ('Anger', 'Anticipation', 'Disgust', 'Fear', 'Joy', 'Sadness', 'Surprise', 'Trust')
    val_corpus.set_label_title(title)
    print(val_corpus.eval())
    print("OK")
