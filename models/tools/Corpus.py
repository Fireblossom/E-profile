import models.tools.Stemming
from nltk.corpus import stopwords
import models.tools.gen_label
from nltk.tokenize import TweetTokenizer
import string
from prettytable import PrettyTable
import nltk


class Corpus:
    def __init__(self, filename):
        self.label_title = []
        with open(filename) as file:
            self.text, self.gold, self.pred = [], [], []
            tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)
            for line in file:
                if line != 'XXXXXXXXXXXX EMPTY ANNOTATION\n':
                    self.gold.append(models.tools.gen_label.gen_label(line))
                    content = line.split('\t')
                    tokens = tknzr.tokenize(content[-1])
                    self.text.append(text(normalization(tokens)))

    def set_predict(self, predict):
        self.pred = predict

    def set_label_title(self, title):
        if len(title) == len(self.gold[0]):
            self.label_title = title
        else:
            print('Titles number error.')

    def eval(self):
        table = PrettyTable(['Label', 'TP', 'FP', 'FN', 'Recall', 'Precision', 'F-Score'])
        if self.pred and self.label_title:
            res = score(self.pred, self.gold)
            for label in range(len(self.label_title)):
                table.add_row([self.label_title[label],
                               res['TP'][label],
                               res['FP'][label],
                               res['FN'][label],
                               round(res['Recall'][label], 4),
                               round(res['Precision'][label], 4),
                               round(res['F-score'][label], 4)
                               ])
            table.add_row(['', '', '', '', '', 'Macro', round(sum(res['F-score'])/len(res['F-score']), 4)])
            return table
        else:
            return False


WANT_TAGS = {'JJ', 'JJR', 'JJS'}
NEG_ADJ = {'wrong', 'bad', 'last', 'dear', 'NOT_equal', 'dead', 'illegal', 'stupid', 'serious', 'worse'}
POS_ADJ = {'good', 'great', 'best', 'happy', 'greatest', 'beautiful', 'agree', 'amazing', 'welcome', 'clear', 'awesome',
           'equal', 'right', 'brilliant', 'dangerous', 'excited', 'nice', 'responsible', 'honest'}


class text:
    def __init__(self, words):
        self.words = [x[0] for x in words]
        self.pos_tags = [x[1] for x in words]

    def feature_extraction(self):
        features = []
        if 'hit' in self.words:
            features.append(1)
        else:
            features.append(0)
        # Rules of feature extraction.
        f_1, f_2, f_3 = 0, 0, 0
        for word in self.words:
            if word in NEG_ADJ:
                f_1 += 1
            if word in POS_ADJ:
                f_2 += 1
        for tag in self.pos_tags:
            if tag in WANT_TAGS:
                f_3 += 1
        features += [f_1, f_2, f_3]
        return features


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
    pos_tags = nltk.pos_tag(tokens)
    for token in pos_tags:
        # token[0] = token[0].replace('\n', '')
        # token[0] = token[0].lower()
        if token[0] == 'not' or token[0][-3:] == "n't":
            negative_word_flag = True
        elif token[0] in string.punctuation:
            negative_word_flag = False

        if token[0] in skip_word_list:
            continue
        elif len(token[0]) >= 8 and token[0][0:8] == 'https://':
            continue  # Skip URL
        elif len(token[0]) >= 1 and token[0][0] == '@':
            continue  # Skip ID
        elif len(token[0]) >= 1 and token[0][0] == '#':
            continue  # Skip #
        elif negative_word_flag:
            # The Porter Algorithm https://tartarus.org/martin/PorterStemmer/python.txt
            # Just fine tuned the code from Python 2 to Python 3.
            terms.append(('NOT_' + token[0].lower(), token[1]))
            # Add NOT symbol until next punctuation. -- following the Stanford NLP.
        else:
            terms.append((token[0].lower(), token[1]))  # p.stem(token, 0, len(token) - 1))

    return terms


def score(predict, gold):
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

    score_dict = {
        'TP':correct,
        'FP':(numpy.array(col_sum(predict)) - numpy.array(correct)).tolist(),
        'FN':(numpy.array(col_sum(gold)) - numpy.array(correct)).tolist(),
        'Recall':recall,
        'Precision':precision,
        'F-score':f_score
    }
    return score_dict


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
