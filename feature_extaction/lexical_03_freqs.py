import nltk
import re
from nltk.tokenize import TweetTokenizer
tknzr = TweetTokenizer(strip_handles=True)

address = r'E:\Uni-Stuttgart\2nd\Computational Linguistics Team Laboratory\datasets\emotions\ssec\extract manual feature'
label = ['Anger', 'Anticipation', 'Disgust', 'Fear', 'Joy', 'Sadness', 'Surprise', 'Trust']


# Calculate the frequency of words that appear only when labeled with a specific word, such as "Anger"
for i in range (len(label)):
        index = label[i]
        dict_with = {}

        with open(address + '\\1 vs 7\\' + index + '.txt', 'r+') as file_with:
                read_with = []
                for line in file_with:
                        read_with += tknzr.tokenize(line)
                        
        with open(address + '\\1 vs 7\\' + 'without_' + index + '.txt', 'r+') as file_without:
                read_without = []
                for line in file_without:
                        read_without += tknzr.tokenize(line)
        #print(read_with[0:140])
        
        for words in read_with:
            if words not in read_without:
                if words not in dict_with:
                        if words != ' ':
                                dict_with[words] = 1
                else:
                        if words != ' ':
                                dict_with[words] += 1
        dict_with = sorted(dict_with.items(), key=lambda d: d[1], reverse=True)
        str_with = ''
        for n in range(len(dict_with)):                   
            str_with += dict_with[n][0] + '\t' + str(dict_with[n][1]) + '\n'

        file_name = open(address + '\\1 vs 7\\' + index + '_freqs.txt', 'r+')
        print(file_name)
        file_name.write(str_with)


for i in range (len(label)):
        index = label[i]
        with open(address + '\\1 vs 7\\' + index + '_freqs.txt', 'r') as r:
            lines = r.readlines()
        with open(address + '\\1 vs 7\\' + index + '_freqs.txt', 'w') as w:
            for n in lines:
                if len(n) > 3:
                    w.write(n)
                    


#Calculates the frequency of words that are expected to appear only if labeled with two specific words, such as "Anger_Anticipation"
for i in range (len(label)):
    if i+1 > 7:
         break
    else:
        for j in range (len(label)):
            if j > i:
                file_name = label[i] + '_' + label[j]
                index = file_name
                dict_with = {}
        
                with open(address + '\\2 vs 6\\' + index + '.txt', 'r+') as file_with:
                        read_with = []
                        for line in file_with:
                                read_with += tknzr.tokenize(line)
                        
                with open(address + '\\2 vs 6\\' + 'without_' + index + '.txt', 'r+') as file_without:
                        read_without = []
                        for line in file_without:
                                read_without += tknzr.tokenize(line)

                for words in read_with:
                    if words not in read_without:
                        if words not in dict_with:
                                if words != ' ':
                                        dict_with[words] = 1
                        else:
                                if words != ' ':
                                        dict_with[words] += 1
                dict_with = sorted(dict_with.items(), key=lambda d: d[1], reverse=True)
                str_with = ''
                for n in range(len(dict_with)):                   
                      str_with += dict_with[n][0] + '\t' + str(dict_with[n][1]) + '\n'

                file_name = open(address + '\\2 vs 6\\' + index + '_freqs.txt', 'r+')
                print(file_name)
                file_name.write(str_with)


for i in range (len(label)):
    if i+1 > 7:
         break
    else:
        for j in range (len(label)):
            if j > i:
                file_name = label[i] + '_' + label[j]
                index = file_name
                with open(address + '\\2 vs 6\\' + index + '_freqs.txt', 'r') as r:
                    lines = r.readlines()
                with open(address + '\\2 vs 6\\' + index + '_freqs.txt', 'w') as w:
                    for n in lines:
                        if len(n) > 3:
                            w.write(n)
                    


#Calculates the frequency of words that are expected to appear only if labeled with three specific words, such as "Anger_Anticipation"
for i in range (len(label)):
    if i+1 > 7:
         break
    else:
        for j in range (len(label)):
            if j > i:
                for q in range (len(label)):
                    if q > j:
                        file_name = label[i] + '_' + label[j] + '_' + label[q]
                        index = file_name
                        dict_with = {}
        
                        with open(address + '\\3 vs 5\\' + index + '.txt', 'r+') as file_with:
                                read_with = []
                                for line in file_with:
                                        read_with += tknzr.tokenize(line)
                        
                        with open(address + '\\3 vs 5\\' + 'without_' + index + '.txt', 'r+') as file_without:
                                read_without = []
                                for line in file_without:
                                        read_without += tknzr.tokenize(line)

        
                        for words in read_with:
                            if words not in read_without:
                                if words not in dict_with:
                                        if words != ' ':
                                                dict_with[words] = 1
                                else:
                                        if words != ' ':
                                                dict_with[words] += 1
                        dict_with = sorted(dict_with.items(), key=lambda d: d[1], reverse=True)
                        str_with = ''
                        for n in range(len(dict_with)):                   
                              str_with += dict_with[n][0] + '\t' + str(dict_with[n][1]) + '\n'

                        file_name = open(address + '\\3 vs 5\\' + index + '_freqs.txt', 'r+')
                        print(file_name)
                        file_name.write(str_with)



for i in range (len(label)):
    if i+1 > 7:
         break
    else:
        for j in range (len(label)):
            if j > i:
                for q in range (len(label)):
                    if q > j:
                        file_name = label[i] + '_' + label[j] + '_' + label[q]
                        index = file_name
                        with open(address + '\\3 vs 5\\' + index + '_freqs.txt', 'r') as r:
                            lines = r.readlines()
                        with open(address + '\\3 vs 5\\' + index + '_freqs.txt', 'w') as w:
                            for n in lines:
                                if len(n) > 3:
                                    w.write(n)


