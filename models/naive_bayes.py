import models.tools.Stemming
import math


def normalization(token):
    """
    input a token and normalize it.
    using a homemade stop word list :P
    :param token:
    :return:
    """
    token = token.replace('\n', '')
    token = token.lower()
    skip_word_list = {'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by',
                      'for', 'from', 'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on',
                      'that', 'the', 'to', 'was', 'were', 'will', 'with', 'so', 'still',
                      'then', 'would', 'ein', 'eine', 'einem', 'einer', 'einen', 'eines',
                      'und', 'sein', 'bin', 'bist', 'ist', 'sind', 'seid', 'war', 'warst',
                      'war', 'waren', 'wart', 'da', 'in', 'ins', 'im', 'an', 'am', 'auf', 'bei',
                      'für', 'hab', 'habe', 'hat', 'haben', 'ich', 'du', 'er', 'es', 'sie',
                      'Sie', 'wir', 'ihr', 'mir', 'dir', 'ihr', 'ihm', 'uns', 'euch', 'ihnen',
                      'mich', 'dich', 'ihn', 'diese', 'dieser', 'der', 'die', 'das', 'zu',
                      'werden', 'werde', 'mit', 'als', 'hier', 'dass', 'so', 'noch', 'dann',
                      'von', 'vom', 'ja'}
    # Remove some grammatical vocabulary in German and English
    if token in skip_word_list:
        return ''
    elif len(token) >= 8 and token[0:8] == 'https://':
        return ''  # Skip URL
    elif len(token) >= 1 and token[0] == '@':
        return ''  # Skip ID
    else:
        if len(token) >= 2 and token[0] == '#':
            token = token[1:]
        # The Porter Algorithm https://tartarus.org/martin/PorterStemmer/python.txt
        # Just fine tuned the code from Python 2 to Python 3.
        p = models.tools.Stemming.PorterStemmer()
        term = p.stem(token, 0, len(token) - 1)
    return term


def list_to_dict(lis, flag):
    """
    count the list elements and record them in the dictionary.
    :param lis:
    :param flag: normalize or not
    :return: the dictionary of
    """
    dic = {}
    for elem in lis:
        e = elem
        if flag:
            e = normalization(elem)
        if e != '':
            if e in dic:
                dic[e] += 1
            else:
                dic[e] = 1
    return dic


def p_tc(dic, term, term_count, b):
    return (dic[term] + 1) / (term_count + b)


def cmap(pc, p_tc, p_lc):
    """
    Classification rule
    :param pc:
    :param p_tc:
    :param p_lc:
    :return:
    """
    return math.log10(pc) + sum(p_tc) + p_lc


class NB_classifier:
    def __init__(self, features):
        self.model = []
        self.features = features

    def train(self, train_file):
        """
        input a string of file and train the model.
        :param train_file:
        :return:
        """
        import models.tools.gen_label
        label = []
        text = []
        for line in train_file:
            label.append(models.tools.gen_label.gen_label(line))
            content = line.split('\t')
            text.append(content[-1].split(" "))  # Try nltk?

        for i in range(self.features):
            positive = []
            positive_count = 0
            negative = []
            negative_count = 0
            for t in range(len(text)):
                if label[t][i] == 1:
                    positive += text[t]
                    positive_count += 1
                else:
                    negative += text[t]
                    negative_count += 1

            positive_dict = list_to_dict(positive, True)
            positive_term = sum(positive_dict.values())
            pc_positive = positive_count / positive_count + negative_count
            negative_dict = list_to_dict(negative, True)
            negative_term = sum(negative_dict.values())
            pc_negative = negative_count / positive_count + negative_count
            # prob of class.

            bins = len(set(list(positive_dict.keys()) + list(negative_dict.keys())))

            p_tc_positive = {}
            for elem in positive_dict:
                p_tc_positive[elem] = p_tc(positive_dict, elem, positive_term, bins)
            p_tc_negative = {}
            for elem in negative_dict:
                p_tc_negative[elem] = p_tc(negative_dict, elem, negative_term, bins)
            # prob of class given term.

            p_lc_positive = {}
            p_lc_negative = {}
            lc_positive = []
            lc_negative = []
            lab = tuple(label)
            for la in lab:
                if la[i] == 1:
                    lc_positive.append(tuple(la[:i]))
                else:
                    lc_negative.append(tuple(la[:i]))

            lc_positive_dict = list_to_dict(lc_positive, False)
            lc_positive_term = sum(lc_positive_dict.values())
            lc_negative_dict = list_to_dict(lc_negative, False)
            lc_negative_term = sum(lc_negative_dict.values())

            lc_bins = len(set(list(lc_positive_dict.keys()) + list(lc_negative_dict.keys())))

            for elem in lc_positive_dict:
                p_lc_positive[elem] = p_tc(lc_positive_dict, elem, lc_positive_term, lc_bins)
            for elem in lc_negative_dict:
                p_lc_negative[elem] = p_tc(lc_negative_dict, elem, lc_negative_term, lc_bins)
            # prob of class given previous labels sequence.

            dic = {'p_tc_positive': p_tc_positive, 'p_tc_negative': p_tc_negative, 'pc_positive': pc_positive,
                   'pc_negative': pc_negative, 'bins': bins, 'positive_term': positive_term,
                   'negative_term': negative_term, 'p_lc_positive': p_lc_positive, 'p_lc_negative': p_lc_negative,
                   'lc_positive_term': lc_positive_term, 'lc_negative_term': lc_negative_term, 'lc_bins': lc_bins}
            self.model.append(dic)  # Training complete.

    def predict(self, text):
        """
        input a string of text and return the predict label of given text.
        :param text:
        :return:
        """
        predict_label = []
        word_list = text.split()
        for i in range(self.features):
            p_tc_positive = []
            p_tc_negative = []
            for word in word_list:
                w = normalization(word)
                try:
                    p_tc_positive.append(math.log10(self.model[i]['p_tc_positive'][w]))
                except KeyError:
                    p_tc_positive.append(math.log10(1 / (self.model[i]['positive_term'] + self.model[i]['bins'])))
                try:
                    p_tc_negative.append(math.log10(self.model[i]['p_tc_negative'][w]))
                except KeyError:
                    p_tc_negative.append(math.log10(1 / (self.model[i]['negative_term'] + self.model[i]['bins'])))
            try:
                p_lc_positive = math.log10(self.model[i]['p_lc_positive'][tuple(predict_label)])
            except KeyError:
                p_lc_positive = math.log10(1 / self.model[i]['lc_positive_term'] + self.model[i]['lc_bins'])
            try:
                p_lc_negative = math.log10(self.model[i]['p_lc_negative'][tuple(predict_label)])
            except KeyError:
                p_lc_negative = math.log10(1 / self.model[i]['lc_negative_term'] + self.model[i]['lc_bins'])

            if cmap(self.model[i]['pc_positive'], p_tc_positive, p_lc_positive) > cmap(self.model[i]['pc_negative'], p_tc_negative, p_lc_negative):
                predict_label.append(1)
            else:
                predict_label.append(0)
            
        return predict_label
