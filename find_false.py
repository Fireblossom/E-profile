def find_fp(corpus):
    """
    返回8个列表，每个列表里是对应标签FP所有的词，可以用Counter快捷计算词频
    :param corpus:
    :return:
    """
    idx = [[]] * 8
    for i in range(len(corpus.gold)):
        for j in range(len(corpus.gold[0])):
            if corpus.gold[i][j] == 0 and corpus.pred[i][j] == 1:
                idx[j].append(i)
    texts = [[]] * 8
    for j in range(len(idx)):
        for i in idx[j]:
            texts[j] += corpus.text[i].words
    return texts


def find_fn(corpus):
    idx = [[]] * 8
    for i in range(len(corpus.gold)):
        for j in range(len(corpus.gold[0])):
            if corpus.gold[i][j] == 1 and corpus.pred[i][j] == 0:
                idx[j].append(i)
    texts = [[]] * 8
    for j in range(len(idx)):
        for i in idx[j]:
            texts[j] += corpus.text[i].words
    return texts
