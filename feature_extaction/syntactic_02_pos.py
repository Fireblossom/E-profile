import nltk
from nltk.corpus import stopwords
import string
from nltk.tokenize import TweetTokenizer
tknzr = TweetTokenizer(strip_handles=True)
label2 = ['Anger', 'Anticipation', 'Disgust', 'Fear', 'Joy', 'Sadness', 'Surprise', 'Trust']
label = ['Trust']

for i in range(len(label)):
    sentences = []
    with open(label[i] + '.txt') as file:
        for line in file:
            sentences.append(line)
        print(len(sentences))


    to_be_write = ''
    n = 0
    for sentence in sentences:
        #print(sentence)
        tokens = tknzr.tokenize(sentence)

        word_and_pos_tags = nltk.pos_tag(tokens)
        #print(word_and_pos_tags)   
        pos_list = []
        for token in word_and_pos_tags:
            pos_list.append(token[1])
        pos_list = ' '.join(pos_list)
        print(pos_list)
        to_be_write += pos_list + '\n'
        n += 1
    print(n)


    file = open('n-gram/' + label[i] + '_pos.txt', 'r+')
    file.writelines(to_be_write)
       



'''
    terms = []
    negative_word_flag = False
    skip_word_list = set(stopwords.words('english'))
'''

'''
    for token in word_and_pos_tags:
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
            terms.append(('NOT_' + token[0].lower(), token[1]))
        else:
            terms.append((token[0].lower(), token[1]))
    to_be_write += terms

    
file = open('test.txt', 'w')
for tup in to_be_write:
    file.write(tup[0] + '/' + tup[1] + '\n')
file.close()
'''
