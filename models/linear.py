import numpy as np
from numpy import dot


class L_classifier:
    def __init__(self, labels):
        self.labels = labels
        self.w = []

    def train(self, corpus):
        for label in range(self.labels):
            labels = []
            vectors = []
            for inst in range(len(corpus.text)):
                feature = corpus.text[inst].feature_extraction(corpus.top_words)
                vectors.append(feature2vector(feature))
                labels.append(corpus.gold[inst][label])
            self.w.append(perceptron(vectors, labels))

    def predict(self, corpus, top_words):
        predict = []
        for inst in range(len(corpus.text)):
            labels = []
            for label in range(self.labels):
                if dot(self.w[label], feature2vector(corpus.text[inst].feature_extraction(top_words))) >= 0:
                    labels.append(1)
                else:
                    labels.append(0)
            predict.append(labels)
        return predict


def perceptron(vectors, labels):
    w = np.array([0] * len(vectors[0]))
    flag = False
    count = 0
    while not flag and count < 100:
        for i in range(len(vectors)):
            t = dot(vectors[i], w)
            if t <= 0 and labels[i] == 1:
                w += vectors[i]
                flag = False
                break
            if t >= 0 and labels[i] == 0:
                w -= vectors[i]
                flag = False
                break
            flag = True
        count += 1
    return w


def feature2vector(feature):
    vector = []
    for elem in feature:
        if feature[elem] is True:
            vector.append(1)
        else:
            vector.append(-1)
    return vector

