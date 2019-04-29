import models.tools.Stemming
import math


def normalization(token):
    import re
    token = re.sub(r'[^\w\s]', '', token)
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


def list_to_dict(lis):
    dic = {}
    for elem in lis:
        e = normalization(elem)
        if e != '':
            if e in dic:
                dic[e] += 1
            else:
                dic[e] = 1
    return dic


def p_tc(dic, term, term_count, b):
    return (dic[term] + 1) / (term_count + b)


def cmap(pc, p_tc):
    return math.log10(pc) + sum(p_tc)


class NB_classifier:
    def __init__(self, features):
        self.model = []
        self.features = features

    def train(self, train_file):
        import models.tools.gen_label
        label = []
        text = []
        for line in train_file:
            label.append = models.tools.gen_label.gen_label(line)
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

            positive_dict = list_to_dict(positive)
            positive_term = sum(positive_dict.values())
            pc_positive = positive_count / positive_count + negative_count
            negative_dict = list_to_dict(negative)
            negative_term = sum(negative_dict.values())
            pc_negative = negative_count / positive_count + negative_count

            bins = len(set(list(positive_dict.keys()) + list(negative_dict.keys())))

            p_tc_positive = {}
            for elem in positive_dict:
                p_tc_positive[elem] = p_tc(positive_dict, elem, positive_term, bins)
            p_tc_negative = {}
            for elem in negative_dict:
                p_tc_negative[elem] = p_tc(negative_dict, elem, negative_term, bins)
            
            dic = {'p_tc_positive': p_tc_positive, 'p_tc_negative': p_tc_negative, 'pc_positive': pc_positive,
                   'pc_negative': pc_negative, 'bins': bins, 'positive_term': positive_term,
                   'negative_term': negative_term}
            self.model.append(dic)  # Training complete.

    def predict(self, text):
        predict_label = []
        word_list = text.split()
        for i in range(self.features):
            p_tc_positive = []
            p_tc_negative = []
            for word in word_list:
                w = normalization(word)
                try:
                    p_tc_positive.append(math.log10(self.model[i]['p_tc_positive'][w]))
                except:
                    p_tc_positive.append(1 / (self.model[i]['positive_term'] + self.model[i]['bins']))
                try:
                    p_tc_negative.append(math.log10(self.model[i]['p_tc_negative'][w]))
                except:
                    p_tc_negative.append(1 / (self.model[i]['negative_term'] + self.model[i]['bins']))
                    
            if cmap(self.model[i]['pc_positive'], p_tc_positive) > cmap(self.model[i]['pc_negative'], p_tc_negative):
                predict_label.append(True)
            else:
                predict_label.append(False)
            
        return predict_label
