def write_file(label_list, label_title, text, filename):
    """
    input the predict label list , the title of each emotional label, text and file name
    write those information into the last line of predict file.
    :param label_list:
    :param label_title:
    :param text:
    :param filename:
    :return:
    """
    f = open(filename, 'a')
    write_line = ''
    for label in range(len(label_list)):
        if label_list[label] == 1:
            write_line += label_title[label]
        else:
            write_line += '---'
        write_line += '\t'
    write_line += text
    f.write(write_line)
