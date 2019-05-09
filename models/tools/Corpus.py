import models.tools.Stemming
from nltk.corpus import stopwords
import models.tools.gen_label
from nltk.tokenize import TweetTokenizer
import string


class Corpus:
    def __init__(self, filename):
        with open(filename) as file:
            self.text, self.gold, self.pred = [], [], []
            tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)
            for line in file:
                if line != 'XXXXXXXXXXXX EMPTY ANNOTATION\n':
                    self.gold.append(models.tools.gen_label.gen_label(line))
                    content = line.split('\t')
                    self.text.append(normalization(tknzr.tokenize(content[-1])))

    def set_predict(self, predict):
        self.pred = predict

    def eval(self):
        if self.pred:
            f = f_score(self.pred, self.gold)
            return f
        else:
            return False


def normalization(tokens):
    """
    input a token and normalize it.
    using nltk stop word list :P
    :param tokens:
    :return:
    """
    terms = []
    negative_word_flag = False
    skip_word_list = set(stopwords.words('english'))
    # Remove some grammatical vocabulary in English
    p = models.tools.Stemming.PorterStemmer()
    for token in tokens:
        token = token.replace('\n', '')
        token = token.lower()
        if token == 'not' or token[-3:] == "n't":
            negative_word_flag = True
        elif token in string.punctuation:
            negative_word_flag = False

        if token in skip_word_list:
            continue
        elif len(token) >= 8 and token[0:8] == 'https://':
            continue  # Skip URL
        elif len(token) >= 1 and token[0] == '@':
            continue  # Skip ID
        elif negative_word_flag:
            # The Porter Algorithm https://tartarus.org/martin/PorterStemmer/python.txt
            # Just fine tuned the code from Python 2 to Python 3.
            terms.append('NOT_' + p.stem(token, 0, len(token) - 1))
            # Add NOT symbol until next punctuation. -- following the Stanford NLP.
        else:
            terms.append(p.stem(token, 0, len(token) - 1))

    return terms


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
