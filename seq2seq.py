from __future__ import print_function
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import sys

import argparse
from seq2seq_utils import *

ap = argparse.ArgumentParser()
ap.add_argument('-max_len', type=int, default=200)
ap.add_argument('-vocab_size', type=int, default=20000)
ap.add_argument('-batch_size', type=int, default=100)
ap.add_argument('-layer_num', type=int, default=3)
ap.add_argument('-hidden_dim', type=int, default=1000)
ap.add_argument('-nb_epoch', type=int, default=20)
ap.add_argument('-mode', default='train')
args = vars(ap.parse_args())

MAX_LEN = args['max_len']
VOCAB_SIZE = args['vocab_size']
BATCH_SIZE = args['batch_size']
LAYER_NUM = args['layer_num']
HIDDEN_DIM = args['hidden_dim']
NB_EPOCH = args['nb_epoch']
MODE = args['mode']

if __name__ == '__main__':
    # Loading input sequences, output sequences and the necessary mapping dictionaries
    print('[INFO] Loading data...')
    X, X_vocab_len, X_word_to_ix, X_ix_to_word, y, y_vocab_len, y_word_to_ix, y_ix_to_word = load_data('europarl-v8.fi-en.en', 'europarl-v8.fi-en.fi', MAX_LEN, VOCAB_SIZE)

    # Finding the length of the longest sequence
    X_max_len = max([len(sentence) for sentence in X])
    y_max_len = max([len(sentence) for sentence in y])

    # Padding zeros to make all sequences have a same length with the longest one
    print('[INFO] Zero padding...')
    X = pad_sequences(X, maxlen=X_max_len, dtype='int32')
    y = pad_sequences(y, maxlen=y_max_len, dtype='int32')

    # Creating the network model
    print('[INFO] Compiling model...')
    model = create_model(X_vocab_len, X_max_len, y_vocab_len, y_max_len, HIDDEN_DIM, LAYER_NUM)

    # Finding trained weights of previous epoch if any
    saved_weights = find_checkpoint_file('.')

    # Training only if we chose training mode
    if MODE == 'train':
        k_start = 1

        # If any trained weight was found, then load them into the model
        if len(saved_weights) != 0:
            print('[INFO] Saved weights found, loading...')
            epoch = saved_weights[saved_weights.rfind('_')+1:saved_weights.rfind('.')]
            model.load_weights(saved_weights)
            k_start = epoch + 1

        i_end = 0
        for k in range(k_start, NB_EPOCH+1):
            # Shuffling the training data every epoch to avoid local minima
            indices = np.arange(len(X))
            np.random.shuffle(indices)
            X = X[indices]
            y = y[indices]

            # Training 1000 sequences at a time
            for i in range(0, len(X), 1000):
                if i + 1000 >= len(X):
                    i_end = len(X)
                else:
                    i_end = i + 1000
                y_sequences = process_data(y[i:i_end], y_max_len, y_word_to_ix)

                print('[INFO] Training model: epoch {}th {}/{} samples'.format(k, i, len(X)))
                model.fit(X[i:i_end], y_sequences, batch_size=BATCH_SIZE, nb_epoch=1, verbose=2)
            model.save_weights('checkpoint_epoch_{}.hdf5'.format(k))
    
    # Performing test if we chose test mode
    else:
        # Only performing test if there is any saved weights
        if len(saved_weights) == 0:
            print("The network hasn't been trained! Program will exit...")
            sys.exit()
        else:
            X_test = load_test_data('test', X_word_to_ix, MAX_LEN)
            X_test = pad_sequences(X_test, maxlen=X_max_len, dtype='int32')
            model.load_weights(saved_weights)
            y_pred = np.argmax(model.predict(X_test)[0], axis=1)
            print(y_pred)