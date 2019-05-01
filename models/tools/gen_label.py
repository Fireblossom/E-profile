def gen_label(line):
    """
    generate a list of label, 1 is positive, 0 is negative.
    :param line:
    :return:
    """
    lst = line.split('\t')
    re = []
    for elem in lst:
        if elem != '---':
            re.append(1)
        else:
            re.append(0)
    return re[:-1]
