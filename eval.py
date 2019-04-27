class Eval:
    def __init__(self, predict_filename, gold_filename):
        f = self.f_score(self.lst_gen(predict_filename), self.lst_gen(gold_filename))

    def lst_gen(self, filename):
        """
        Read file to generate binary labels.
        :param filename:
        :return:
        """
        re = []
        with open(filename, 'r') as predict_file:
            for line in predict_file:
                lst = line.split(' ')
                sample = []
                for elem in lst:
                    if elem == '-':
                        sample.append(0)
                    else:
                        sample.append(1)
                del sample[-1]
                re.append(sample)
        return re

    def f_score(self, predict, gold):
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

    def col_sum(self, mat):
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
