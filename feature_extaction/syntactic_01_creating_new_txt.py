label = ['Anger', 'Anticipation', 'Disgust', 'Fear', 'Joy', 'Sadness', 'Surprise', 'Trust']

'''
# create .txt for pos
for i in range(len(label)):
    file_name = label[i] + '_pos.txt'
    f = open('n-gram/'+ file_name, 'w')
    f.close()

# create .txt for without_pos
for i in range(len(label)):
    file_name = 'without_' + label[i] + '_pos.txt'
    f = open('n-gram/'+ file_name, 'w')
    f.close()

# create .txt for n-gram
for i in range(len(label)):
    file_name = label[i] + '_n-gram.txt'
    f = open('n-gram/isolate/'+ file_name, 'w')
    f.close()


# create .txt for without_n-gram
for i in range(len(label)):
    file_name = 'without_' + label[i] + '_n-gram.txt'
    f = open('n-gram/isolate/'+file_name, 'w')
    f.close()
'''

# create .txt for syntax_features
for i in range(len(label)):
    file_name = label[i] + '_syntax.txt'
    f = open('n-gram/isolate/syntax/'+file_name, 'w')
    f.close()
