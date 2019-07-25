address = r'E:\Uni-Stuttgart\2nd\Computational Linguistics Team Laboratory\datasets\emotions\ssec\extract manual feature'

# train.csv including two lines must be deleted at first
train = open(address + '\\' + 'train.csv')
read_lines = train.readlines()
label = ['Anger', 'Anticipation', 'Disgust', 'Fear', 'Joy', 'Sadness', 'Surprise', 'Trust']


# extract data for 1 vs. 7
for i in range (len(label)):    
    index = label[i]

    with open(address + '\\1 vs 7\\' + index + '.txt', 'r+') as file_with:
        for line in read_lines:
            if line == None:
                break
            else:
                line = line.split('\t')
                # (1 vs. 7) Collect tweets with a specific tag, like 'Anger'
                line[i] = line[i].replace('"', '')
                if line[i] == index:
                    #print(len(line[8].split('\t')))
                    if len(line[8].split('\t')) > 0:
                        #print(line[8])
                        file_with.write(line[8])


    with open(address + '\\1 vs 7\\' + 'without_' + label[i] + '.txt', 'r+') as file_without:
          for line in read_lines:
            if line == None:
                break
            else:
                line = line.split('\t')
                # (1 vs. 7) Collect tweets without a specific tag, like 'Anger'
                line[i] = line[i].replace('"', '')
                if line[i] != index:
                    file_without.write(line[8])
          

# extract data for 2 vs. 6
for i in range (len(label)):
    if i+1 > 7:
         break
    else:
        for j in range (len(label)):
            if j > i:
                file_name = label[i] + '_' + label[j]
                index = file_name
                # print(index_list)
                
                with open(address + '\\2 vs 6\\' + index + '.txt', 'r+') as file_with:
                    for line in read_lines:
                        if line == None:
                            break
                        else:
                            line = line.split('\t')
                            # (2 vs. 6) Collect tweets with two specific tags, like 'Anger_Anticipation'
                            line[i] = line[i].replace('"', '')
                            label_combine = line[i] + '_' + line[j] 
                            if label_combine == index:
                                file_with.write(line[8])


                with open(address + '\\2 vs 6\\' + 'without_' + index + '.txt', 'r+') as file_with:
                    for line in read_lines:
                        if line == None:
                            break
                        else:
                            line = line.split('\t')
                            # (2 vs. 6) Collect tweets without two specific tags, like 'Anger_Anticipation'
                            line[i] = line[i].replace('"', '')
                            label_combine = line[i] + '_' + line[j]
                            #print(label_combine)
                            if label_combine == '---_---':
                                 file_with.write(line[8])


# extract data for 3 vs. 5
for i in range (len(label)):
    if i+1 > 7:
         break
    else:
        for j in range (len(label)):
            if j > i:
                for q in range (len(label)):
                    if q > j:
                        print(i,j,q)
                        file_name = label[i] + '_' + label[j] + '_' + label[q]
                        index = file_name
                        # print(index)
                
                        with open(address + '\\3 vs 5\\' + index + '.txt', 'r+') as file_with:
                            #print(file_with)
                            for line in read_lines:
                                if line == None:
                                    break
                                else:
                                    line = line.split('\t')
                                    # (3 vs. 5) Collect tweets with three specific tags, like 'Anger_Anticipation_Disgust'
                                    line[i] = line[i].replace('"', '')
                                    label_combine = line[i] + '_' + line[j] + '_' + line[q]
                                    if label_combine == index:
                                        file_with.write(line[8])


                        with open(address + '\\3 vs 5\\' + 'without_' + index + '.txt', 'r+') as file_with:
                            for line in read_lines:
                                if line == None:
                                    break
                                else:
                                    line = line.split('\t')
                                    # (3 vs. 5) Collect tweets without three specific tags, like 'Anger_Anticipation_Disgust'
                                    line[i] = line[i].replace('"', '')
                                    label_combine = line[i] + '_' + line[j] + '_' + line[q] 
                                    if label_combine == '---_---_---':
                                        file_with.write(line[8])

