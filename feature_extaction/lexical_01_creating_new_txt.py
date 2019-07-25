# Author: Tan

label = ['Anger', 'Anticipation', 'Disgust', 'Fear', 'Joy', 'Sadness', 'Surprise', 'Trust']

'''
# create .txt for 2 vs. 6
for i in range(len(label)):
    if i+1 > 7:
        break
    else:
        for j in range (len(label)):
            if j > i:
                file_name = label[i] + '_' + label[j] + '.txt'
                f = open('2 vs 6/' + file_name, 'w')
                f.close()


for i in range(len(label)):
    if i+1 > 7:
        break
    else:
        for j in range (len(label)):
            if j > i:
                file_name = 'without_' + label[i] + '_' + label[j] + '.txt'
                f = open('2 vs 6/' + file_name, 'w')
                f.close()


for i in range(len(label)):
    if i+1 > 7:
        break
    else:
        for j in range (len(label)):
            if j > i:
                file_name = label[i] + '_' + label[j] + '_freqs' + '.txt'
                f = open('2 vs 6/' + file_name, 'w')
                f.close()


# create .txt for 3 vs. 5
for i in range(len(label)):
    if i+1 > 7:
        break
    else:
        for j in range (len(label)):
            if j > i:
                for q in range (len(label)):
                    if q > j:
                        file_name = label[i] + '_' + label[j] + '_' + label[q] + '.txt'
                        f = open('3 vs 5/' + file_name, 'w')
                        f.close()


for i in range(len(label)):
    if i+1 > 7:
        break
    else:
        for j in range (len(label)):
            if j > i:
                for q in range (len(label)):
                    if q > j:
                        file_name = 'without_' + label[i] + '_' + label[j] + '_' + label[q] + '.txt'
                        f = open('3 vs 5/' + file_name, 'w')
                        f.close()


for i in range(len(label)):
    if i+1 > 7:
        break
    else:
        for j in range (len(label)):
            if j > i:
                for q in range (len(label)):
                    if q > j:
                        file_name = label[i] + '_' + label[j] + '_' + label[q] + '_freqs' + '.txt'
                        f = open('3 vs 5/' + file_name, 'w')
                        f.close()
'''

# create .txt for feature_extraction
for i in range(len(label)):
    if i > 7:
        break
    else:
        file_name = 'features_1 vs 7_' + label[i] + '.txt'
        f = open('features/1 vs 7/' + file_name, 'w')
        f.close()

        file_name = 'features_2 vs 6_' + label[i] + '.txt'
        f = open('features/2 vs 6/' + file_name, 'w')
        f.close()

        file_name = 'features_3 vs 5_' + label[i] + '.txt'
        f = open('features/3 vs 5/' + file_name, 'w')
        f.close()
        
        file_name = 'features_all_' + label[i] + '.txt'
        f = open('features/all/' + file_name, 'w')
        f.close()


                
