import numpy as np
from numpy import dot


class L_classifier:
    def __init__(self, labels):
        self.labels = labels
        self.w = []

    def train(self, corpus):
        for label in range(self.labels):
            labels = []
            features = []
            for inst in range(len(corpus.text)):
                features.append(np.array(corpus.text[inst].feature_extraction()))
                labels.append(corpus.gold[inst][label])
            self.w.append(perceptron(features, labels))

    def predict(self, corpus):
        predict = []
        for inst in range(len(corpus.text)):
            labels = []
            for label in range(self.labels):
                if dot(self.w[label], corpus.text[inst].feature_extraction()) >= 0:
                    labels.append(1)
                else:
                    labels.append(0)
            predict.append(labels)
        return predict


def perceptron(features, labels):
    w = np.array([0] * len(features[0]))
    flag = False
    count = 0
    while not flag and count < 100:
        for i in range(len(features)):
            t = dot(features[i], w)
            if t <= 0 and labels[i] == 1:
                w += features[i]
                flag = False
                break
            if t >= 0 and labels[i] == 0:
                w -= features[i]
                flag = False
                break
            flag = True
        count += 1
    return w
