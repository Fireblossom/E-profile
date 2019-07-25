import math
import keras
import numpy as np
import os
import datetime as dt
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Embedding, Bidirectional, Dense, Input, Dropout, CuDNNLSTM
from keras.models import Model
from keras_ordered_neurons import ONLSTM
from models.tools.Corpus import dict_generator, set_generator


class LSTMModel(object):
    def __init__(self):
        self.model = Model()

    def load_model(self, filepath):
        print('[Model] Loading model from file %s' % filepath)
        return keras.models.load_model(filepath, custom_objects={'ONLSTM': ONLSTM})

    def build_model(self, config):
        main_input = Input(shape=(config['maxlen'],), dtype='int32', name='main_input')
        x = Embedding(input_dim=len(config['word_index']),
                      output_dim=config['veclen'],
                      weights=[config['embedding_matrix']],
                      input_length=config['maxlen'],
                      trainable=False)(main_input)
        x = Bidirectional(ONLSTM(units=64, chunk_size=8, return_sequences=True, recurrent_dropconnect=0.25))(x)
        # x = Bidirectional(CuDNNLSTM(units=64, return_sequences=True))(x)
        x = Dropout(0.2)(x)
        x = Bidirectional(ONLSTM(units=64, chunk_size=8, recurrent_dropconnect=0.25))(x)
        # x = Bidirectional(CuDNNLSTM(units=64))(x)
        lstm_out = Dropout(0.2)(x)
        # lstm_out = Bidirectional(ONLSTM(units=64, chunk_size=4))(x)

        feature_input = Input(shape=(len(config['syntax_features'][0]),), name='feature_input')
        x = keras.layers.concatenate([lstm_out, feature_input])

        x_1 = Dense(units=32, activation='relu')(x)

        main_output_1 = Dense(units=1, activation='sigmoid')(x_1)

        self.model = Model(inputs=[main_input, feature_input],
                           outputs=main_output_1)
        self.model.compile(optimizer='adam', loss=keras.losses.binary_crossentropy, metrics=['accuracy'])

    def train(self, corpus, config, epochs, batch_size, save_dir='train/'):
        print('[Model] Training Started')
        print('[Model] %s epochs, %s batch size' % (epochs, batch_size))
        train_data = prep_x(corpus, config['word_index'])
        x = keras.preprocessing.sequence.pad_sequences(train_data,
                                                       value=0,
                                                       padding='post',
                                                       maxlen=config['maxlen'])

        for i in range(8):
            save_fname = os.path.join(save_dir, str(i), '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'),
                                                                       str(epochs)))
            try:
                os.mkdir(save_dir + str(i) + '/')
            except FileExistsError:
                pass

            callbacks = [
                EarlyStopping(monitor='val_loss', patience=2),
                ModelCheckpoint(filepath=save_fname, monitor='val_loss', save_best_only=True)
            ]
            self.model.reset_states()
            self.model.fit(
                [x, config['syntax_features']],
                prep_y(corpus, i),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks
            )
            self.model.save(save_fname)
            
            print('[Model] Training Completed. Model saved as %s' % save_fname)
        print('[Model] All Training Completed.')

    def predict(self, corpus, config, model_filepath):
        train_data = prep_x(corpus, config['word_index'])
        x = keras.preprocessing.sequence.pad_sequences(train_data,
                                                       value=0,
                                                       padding='post',
                                                       maxlen=config['maxlen'])

        pred = []
        for i in range(8):
            model_path = new_file(model_filepath + '/' + str(i))
            pred_model = self.load_model(model_path)
            pred_model._make_predict_function()
            col = pred_model.predict([x, config['syntax_features']])
            pred.append(col)
        return res_prep(pred)


def prep_x(corpus, word_dict):
    x = []
    # The first indices are reserved
    word_index = {k: (v + 3) for k, v in word_dict.items()}
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2  # unknown
    word_index["<UNUSED>"] = 3

    for text in corpus.text:
        sentence = []
        for word in text.words:
            try:
                sentence.append(word_index[word])
            except KeyError:
                sentence.append(2)
        x.append(sentence)
    return x


def prep_y(corpus, idx=-1):
    """
    preparation of labels, if idx < 0 return all labels.
    :param corpus:
    :param idx:
    :return:
    """
    if idx < 0:
        labels = []
        for i in range(len(corpus.gold[0])):
            label = []
            for elem in corpus.gold:
                label.append(elem[idx])
            labels.append(label)
        return labels
    else:
        label = []
        for elem in corpus.gold:
            label.append(elem[idx])
        return label


def load_vector(vector_file):
    """
    load word embdding file, return the word dictionary and vector matrix.
    :param vector_file: word embdding file
    :return:
    """
    with open(vector_file) as f:
        lines = f.readlines()
        embedding_tuple = [tuple(line.strip().split(' ', 1)) for line in lines]
        embedding_tuple = [(t[0].strip().lower(), list(map(float, t[1].split()))) for t in embedding_tuple]
    vec_dict = dict()
    vec_mat = []
    count = 0
    for word, embedding in embedding_tuple:
        if vec_dict.get(word) is None:
            vec_dict[word] = count
            vec_dict['NOT_' + word] = count + len(embedding_tuple)
            count += 1
            vec_mat.append(np.array(embedding))
    vec_mat = np.append(np.array(vec_mat), -1 * np.array(vec_mat), axis=0)
    return vec_dict, vec_mat


def new_file(dir):
    """
    find newest file in dir.
    :param dir:
    :return:
    """
    list = os.listdir(dir)
    list.sort(key=lambda fn: os.path.getmtime(dir + '/' + fn))
    filepath = os.path.join(dir, list[-1])
    if filepath[-9:] == '.DS_Store':
        filepath = os.path.join(dir, list[-2])
    return filepath


def res_prep(raw_result):
    result = []
    for row in raw_result:
        re_row = []
        for col in row:
            if col[0] >= 0.5:
                re_row.append(1)
            else:
                re_row.append(0)
        result.append(re_row)
    return list(map(list, zip(*result)))


def merge_embdding(wordemb, vecmat, path='features'):
    dicts = dict_generator(path)
    features = np.zeros((vecmat.shape[0], len(dicts)))
    for word in wordemb:
        for i in range(len(dicts)):
            if word not in dicts[i]:
                features[wordemb[word]][i] = -1
            elif dicts[i][word] == 0:
                features[wordemb[word]][i] = -1
            elif dicts[i][word] > 0:
                features[wordemb[word]][i] = math.log(dicts[i][word])
    new_vecmat = np.column_stack([vecmat, features])
    return new_vecmat


def syntax_features(corpus, path='syntax'):
    def subsequences(lst):
        re = set()
        for l in range(3, 8):
            if l > len(lst):
                break
            for i in range(len(lst) - l):
                re.add(tuple(lst[i:i + l]))
        return re
    sets = set_generator(path)
    features = np.zeros((len(corpus.text), len(sets)))
    j = 0
    for sentence in corpus.pos:
        for i in range(len(sets)):
            if len(subsequences(sentence).intersection(sets[i])) != 0:
                features[j][i] = math.log2(len(subsequences(sentence).intersection(sets[i])))
            else:
                features[j][i] = -1
        j += 1
    return features


