import nltk
import re
from nltk.tokenize import TweetTokenizer
tknzr = TweetTokenizer(strip_handles=True)
label = ['Anger', 'Anticipation', 'Disgust', 'Fear', 'Joy', 'Sadness', 'Surprise', 'Trust']


for i in range (len(label)):
        index = label[i]
        dict_with = {}

        with open('n-gram/isolate/' + index + '_n-gram.txt', 'r+') as file_with:
            read_with = {}
            for line in file_with:
                a = line.split('\t')[3]
                read_with[a] = line.split('\t')[1]
                #print(read_with)
                        
                       
        with open('n-gram/isolate/' + 'without_' + index + '_n-gram.txt', 'r+') as file_without:
            read_without = {}
            for line in file_without:
                b = line.split('\t')[3]
                read_without[b] = 1
                #print(read_without)

   
        for pos, freqs in read_with.items():
            if pos not in read_without:
                dict_with[pos] = freqs
                #print(dict_with)



        file_name = open('n-gram/isolate/syntax/' + index + '_syntax.txt', 'r+')
        for key, value in dict_with.items():
            write_input = value + '\t' + key + '\n'    
            file_name.write(write_input)

'''
for i in range (len(label)):
        index = label[i]
        with open(address + 'n-gram/isolate/' + index + '_freqs.txt', 'r') as r:
            lines = r.readlines()
        with open(address + 'n-gram/isolate/' + index + '_freqs.txt', 'w') as w:
            for n in lines:
                if len(n) > 3:
                    w.write(n)
'''
