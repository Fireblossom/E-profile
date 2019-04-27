def gen_label(line):
    lst = line.split('\t')
    re = []
    for elem in lst:
        if elem != '-':
            re.append(True)
        else:
            re.append(False)
    return re[:-1]
