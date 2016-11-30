from __future__ import print_function
from keras.models import Sequential
from keras.layers import Activation, TimeDistributed, Dense, RepeatVector, recurrent
from keras.layers.recurrent import LSTM
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import text_to_word_sequence
import numpy as np
from nltk import FreqDist
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-max_len', type=int, default=200)
ap.add_argument('-vocab_size', type=int, default=10000)
ap.add_argument('-batch_size', type=int, default=10)
ap.add_argument('-layer_num', type=int, default=3)
ap.add_argument('-hidden_dim', type=int, default=500)
ap.add_argument('-nb_epoch', type=int, default=20)
ap.add_argument('-weights', default='')
args = vars(ap.parse_args())

MAX_LEN = args['max_len']
VOCAB_SIZE = args['vocab_size']
BATCH_SIZE = args['batch_size']
LAYER_NUM = args['layer_num']
HIDDEN_DIM = args['hidden_dim']
NB_EPOCH = args['nb_epoch']
WEIGHTS = args['weights']

def load_data(source, dist, max_len, vocab_size):
    f = open(source, 'r')
    X_data = f.read()
    f.close()
    f = open(dist, 'r')
    y_data = f.read()
    f.close()

    X = [text_to_word_sequence(x) for x, y in zip(X_data.split('\n'), y_data.split('\n')) if len(x) <= max_len and len(y) <= max_len]
    y = [text_to_word_sequence(y) for x, y in zip(X_data.split('\n'), y_data.split('\n')) if len(x) <= max_len and len(y) <= max_len]

    dist = FreqDist(np.hstack(X))
    X_vocab = dist.most_common(vocab_size-1)
    dist = FreqDist(np.hstack(y))
    y_vocab = dist.most_common(vocab_size-1)

    X_ix_to_word = [word[0] for word in X_vocab]
    X_ix_to_word.append('UNK')
    X_word_to_ix = {word:ix for ix, word in enumerate(X_ix_to_word)}
    for i, sentence in enumerate(X):
        for j, word in enumerate(sentence):
            if word in X_word_to_ix:
                X[i][j] = X_word_to_ix[word]
            else:
                X[i][j] = X_word_to_ix['UNK']

    y_ix_to_word = [word[0] for word in y_vocab]
    y_ix_to_word.append('UNK')
    y_word_to_ix = {word:ix for ix, word in enumerate(y_ix_to_word)}
    for i, sentence in enumerate(y):
        for j, word in enumerate(sentence):
            if word in y_word_to_ix:
                y[i][j] = y_word_to_ix[word]
            else:
                y[i][j] = y_word_to_ix['UNK']
    return (X, len(X_vocab)+1, X_word_to_ix, X_ix_to_word, y, len(y_vocab)+1, y_word_to_ix, y_ix_to_word)


def process_data(word_sentences, max_len, word_to_ix):
    sequences = np.zeros((len(word_sentences), max_len, len(word_to_ix)))
    for sentence in word_sentences:
        for i, word in enumerate(sentence):
            sequences[:, i, word] = 1.
    return sequences

def create_model(X_vocab_len, X_max_len, y_vocab_len, y_max_len, hidden_size, num_layers):
    model = Sequential()
    model.add(LSTM(hidden_size, input_shape=(X_max_len, X_vocab_len)))
    model.add(RepeatVector(y_max_len))
    for _ in range(num_layers):
        model.add(LSTM(hidden_size, return_sequences=True))
    model.add(TimeDistributed(Dense(y_vocab_len)))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
            optimizer='rmsprop',
            metrics=['accuracy'])
    return model

if __name__ == '__main__':
    
    X, X_vocab_len, X_word_to_ix, X_ix_to_word, y, y_vocab_len, y_word_to_ix, y_ix_to_word = load_data('europarl-v8.fi-en.en', 'europarl-v8.fi-en.fi', MAX_LEN, VOCAB_SIZE)
    
    X_max_len = max([len(sentence) for sentence in X])
    y_max_len = max([len(sentence) for sentence in y])

    print('[INFO] Compiling model...')
    model = create_model(X_vocab_len, X_max_len, y_vocab_len, y_max_len, HIDDEN_DIM, LAYER_NUM)
    for k in range(1, NB_EPOCH+1):
        for i in range(0, len(X), 100):
            padded_X = pad_sequences(X[i:i+100], maxlen=X_max_len, dtype='uint8')

            padded_y = pad_sequences(y[i:i+100], maxlen=y_max_len, dtype='uint8')

            X_sequences = process_data(padded_X, X_max_len, X_word_to_ix)

            y_sequences = process_data(padded_y, y_max_len, y_word_to_ix)

            print('[INFO] Training model: epoch {}th {}/{} samples'.format(k, i, len(X)))
            model.fit(X_sequences, y_sequences, batch_size=BATCH_SIZE, nb_epoch=1, verbose=2)
        model.save_weights('checkpoint_epoch_{}.hdf5'.format(k))