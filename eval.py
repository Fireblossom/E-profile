import models.tools.gen_label


class Eval:
    def __init__(self, predict_filename, gold_filename):
        pred, gold = [], []
        with open(predict_filename) as predict_file:
            for line in predict_file:
                pred.append(models.tools.gen_label.gen_label(line))
        with open(gold_filename) as gold_file:
            for line in gold_file:
                gold.append(models.tools.gen_label.gen_label(line))

        self.f = f_score(pred, gold)

    def get_score(self):
        return self.f


def f_score(predict, gold):
    """
    Input binary labels output F-score.
    :param predict: list
    :param gold: list
    :return:
    """
    correct = [0] * len(gold[0])
    for i, j in zip(predict, gold):
        for elem in range(len(i)):
            if i[elem] == 1 and j[elem] == 1:
                correct[elem] += 1
    import numpy
    precision = numpy.divide(numpy.array(correct), numpy.array(col_sum(predict))).tolist()
    recall = numpy.divide(numpy.array(correct), numpy.array(col_sum(gold))).tolist()

    f_score = []
    for p, r in zip(precision, recall):
        f_score.append(2 * p * r /( p + r ))
    return f_score


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
