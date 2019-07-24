address = r'E:\Uni-Stuttgart\2nd\Computational Linguistics Team Laboratory\datasets\emotions\ssec\extract manual feature'
label = ['Anger', 'Anticipation', 'Disgust', 'Fear', 'Joy', 'Sadness', 'Surprise', 'Trust']

file_name_1vs7 = []
file_name_2vs6 = []
file_name_3vs5 = []
file_name = []
for i in range(len(label)):
    if i+1 > 7:
         break
    else:
        file_name_1vs7.append('\\1 vs 7\\'+ label[i])
        file_name.append('\\1 vs 7\\'+ label[i])
        for j in range(len(label)):
            if j > i:
                file_name_2vs6.append('\\2 vs 6\\'+ label[i] + '_' + label[j])
                file_name.append('\\2 vs 6\\'+ label[i] + '_' + label[j])
                for q in range(len(label)):
                    if q > j:
                        file_name_3vs5.append('\\3 vs 5\\'+ label[i] + '_' + label[j] + '_' + label[q])
                        file_name.append('\\3 vs 5\\'+ label[i] + '_' + label[j] + '_' + label[q])


for name in file_name_1vs7:
    for i in range(len(label)):
        if label[i] in name:
            f = open(address + name + '_freqs.txt', 'r+').readlines()
            w = open(address + '\\features\\1 vs 7\\' + 'features_1 vs 7_' + label[i] + '.txt', 'a')
            w.writelines(f)

for name in file_name_2vs6:
    for i in range(len(label)):
        if label[i] in name:
            f = open(address + name + '_freqs.txt', 'r+').readlines()
            w = open(address + '\\features\\2 vs 6\\' + 'features_2 vs 6_' + label[i] + '.txt', 'a')
            w.writelines(f)

for name in file_name_3vs5:
    for i in range(len(label)):
        if label[i] in name:
            f = open(address + name + '_freqs.txt', 'r+').readlines()
            w = open(address + '\\features\\3 vs 5\\' + 'features_3 vs 5_' + label[i] + '.txt', 'a')
            w.writelines(f)

for name in file_name:
    for i in range(len(label)):
        if label[i] in name:
            f = open(address + name + '_freqs.txt', 'r+').readlines()
            w = open(address + '\\features\\all\\' + 'features_all_' + label[i] + '.txt', 'a')
            w.writelines(f)


