import numpy as np


class L_classifier:
    def __init__(self, feature):
        self.feature = feature

    def train(self, corpus):
        pass


def perceptron(W, w1, w2):
    flag = False
    while not flag:
        for i in range(len(w1)):
            t1 = 0
            t2 = 0
            for j in range(len(W)):
                t1 += W[j] * w1[i][j]
                t2 += W[j] * w2[i][j]
            if t1 <= 0:
                for j in range(len(W)):
                    W[j] += w1[i][j]
                flag = False
                break
            if t2 >= 0:
                for j in range(len(W)):
                    W[j] -= w2[i][j]
                flag = False
                break
            flag = True
    return W
