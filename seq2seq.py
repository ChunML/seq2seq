from __future__ import print_function
from keras.models import Sequential
from keras.layers import Activation, TimeDistributed, Dense, RepeatVector, recurrent
from keras.layers.recurrent import LSTM
import numpy as np

def load_data(filename):
    f = open(filename, 'r')
    data = f.read()
    f.close()
    sentences = data.split('\n')
    sentences = sentences[:-1]
    word_sentences = [sentence.split(' ') for sentence in sentences]
    vocab = list(set(np.hstack(word_sentences)))
    word_to_ix = {word:ix for ix, word in enumerate(vocab)}
    ix_to_word = {ix:word for ix, word in enumerate(vocab)}
    return (word_sentences, len(vocab), word_to_ix, ix_to_word)

def process_data(word_sentences, word_to_ix):
    sequences = []
    for sentence in word_sentences:
        sequence = np.zeros((len(sentence), len(word_to_ix)))
        for i, word in enumerate(sentence):
            sequence[i, word_to_ix[word]] = 1
        sequences.append(sequence)
    return sequences

def create_model(source_vocab_len, dest_vocab_len, hidden_size, num_layers):
    model = Sequential()
    model.add(LSTM(hidden_size, input_shape=(None, source_vocab_len)))
    model.add(RepeatVector(8))
    for _ in range(num_layers):
        model.add(LSTM(hidden_size, return_sequences=True))
    model.add(TimeDistributed(Dense(dest_vocab_len)))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])
    return model

if __name__ == '__main__':
    X_sentences, X_vocab_len, X_word_to_ix, X_ix_to_word = load_data('en.txt')
    y_sentences, y_vocab_len, y_word_to_ix, y_ix_to_word = load_data('fr.txt')
    X_max_len = max([len(sentence) for sentence in X_sentences])
    print(X_max_len)
    X_sequences = process_data(X_sentences, X_word_to_ix)
    y_sequences = process_data(y_sentences, y_word_to_ix)
    model = create_model(X_vocab_len, y_vocab_len, 100, 3)
    model.fit(X_sequences[0][np.newaxis, :], y_sequences[0][np.newaxis, :], verbose=1)
    print('Done')
