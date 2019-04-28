import models.tools.gen_label


class Eval:
    def __init__(self, predict_filename, gold_filename):
        self.f = f_score(models.tools.gen_label.gen_label(predict_filename), models.tools.gen_label.gen_label(gold_filename))

    def get_score(self):
        return self.f


def f_score(predict, gold):
    """
    Input binary labels output F-score.
    :param predict: list
    :param gold: list
    :return:
    """
    acc = [0] * len(gold[0])
    for i, j in zip(predict, gold):
        for elem in range(len(i)):
            if i[elem] == j[elem]:
                acc[elem] += 1
    import numpy
    precision = sum(numpy.divide(numpy.array(acc), numpy.array(self.col_sum(predict))).tolist())
    recall = sum(numpy.divide(numpy.array(acc), numpy.array(self.col_sum(gold))).tolist())
    return 2 * precision * recall / (precision + recall)


def col_sum(mat):
    """
    Calculate the sum of the columns.
    :param mat: list
    :param i: index
    :return:
    """
    s = [0] * len(mat[0])
    for row in mat:
        for i in range(len(mat[0])):
            s[i] += row[i]
    return s
