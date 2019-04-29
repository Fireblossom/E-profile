def gen_label(line):
    lst = line.split('\t')
    re = []
    for elem in lst:
        if elem != '---':
            re.append(1)
        else:
            re.append(0)
    return re[:-1]
