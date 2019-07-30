# Author: Duan & Tan

import numpy as np
from numpy import dot


class L_classifier:
    def __init__(self, labels):
        """
        :param labels: number of labels
        """
        self.labels = labels
        self.w = []

    def train(self, corpus, dicts, syntax_feature):
        """
        read corpus, generate features and convert features to vector, then train the hyperplane w.
        :param corpus:
        :return:
        """
        for label in range(self.labels):
            labels = []
            vectors = []
            for inst in range(len(corpus.text)):
                feature = corpus.text[inst].feature_extraction(dicts)
                # print(feature)
                vectors.append(list(syntax_feature[inst])+feature2vector(feature, corpus.tf_idf))
                # print(vector)
                labels.append(corpus.gold[inst][label])
            self.w.append(perceptron(vectors, labels))

    def predict(self, corpus, tf_idf, dicts, syntax_feature):
        """
        read corpus, making predict using the trained w, return labels for all instance in corpus.
        :param corpus:
        :param tf_idf: the tf_idf dictionary of train set used for vector generation
        :return: labels for all instance in corpus.
        """
        predict = []
        for inst in range(len(corpus.text)):
            labels = []
            for label in range(self.labels):
                if dot(self.w[label],
                       list(syntax_feature[inst])+feature2vector(corpus.text[inst].feature_extraction(dicts),
                                      tf_idf)) >= 0:
                    labels.append(1)
                else:
                    labels.append(0)
            predict.append(labels)
        return predict


def perceptron(vectors, labels):
    """
    perceptron for linear model, input some vectors and its gold labels,
    :param vectors: train vectors generate by features.
    :param labels: gold labels
    :return: trained hyperplane w.
    """
    w = np.array([0.0] * len(vectors[0]))
    flag = False
    count = 0
    while not flag and count < 1000:
        for i in range(len(vectors)):
            vec = np.array(vectors[i], dtype=np.float64)
            t = dot(vec, w)
            if t <= 0 and labels[i] == 1:
                w += 0.1 * vec
                flag = False
                break
            if t >= 0 and labels[i] == 0:
                w -= 0.1 * vec
                flag = False
                break
            flag = True
        count += 1
    # print(w)
    return w


def feature2vector(feature, tf_idf):
    """
    Convert boolean, and word features into vector.
    For boolean features, True is +1 and False is -1.
    For word features, set tf_idf weight * the number of occurrences to its dim, otherwise 0.
    :param feature:
    :return:
    """
    vector = []

    for elem in feature[:40]:
        if elem is True:
            vector.append(1.0)
        elif elem is False:
            vector.append(-1.0)
        elif type(elem) != str:
            vector.append(elem)

    for word in tf_idf.keys():
        vector.append(tf_idf[word] * feature.count(word))
    # print(vector)
    return vector

