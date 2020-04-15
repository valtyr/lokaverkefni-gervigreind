from tokenizer import TokenizedData
from helpers import pairs
from data import Results

from keras.layers import Dense, Activation, Embedding, GRU, Conv1D, Dropout, Bidirectional, SpatialDropout1D, Input, Lambda, MaxPooling1D, GlobalAveragePooling1D, concatenate, LSTM
from keras.models import Model as KerasModel, Sequential, load_model
from keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer, text_to_word_sequence

import tensorflow as tf
from keras.layers.core import Flatten
# tf.config.experimental.set_visible_devices([], 'GPU')


class Model:
    def __init__(self):
        self.model = Sequential()

    def build(self, dataset: TokenizedData):
        input_shape = dataset.input.shape
        n_comments, width = input_shape

        layers = [
            Embedding(len(dataset.tokens) + 1, 128),
            #SpatialDropout1D(0.2),
            Bidirectional(LSTM(128, recurrent_dropout=0.2)),
            Dense(128, activation='relu'),
            #SpatialDropout1D(0.2),
            #MaxPooling1D(),
            Dense(32),
            Dropout(0.25),
            Dense(6, activation='softmax'),
        ]
        #input, embedding, spatialdropuot1d globalmaxpooling, concatenate, dropout, dense ,dense

        for layer in layers:
            self.model.add(layer)

        optimizer = Adam(learning_rate=0.01)
        self.model.compile(optimizer=optimizer,
                           loss='binary_crossentropy',
                           metrics=['accuracy'])

    def fit(self, dataset: TokenizedData):
        history = self.model.fit(dataset.input,
                                 dataset.output,
                                 batch_size=64,
                                 verbose=True,
                                 epochs=5,
                                 validation_split=0.4)
        return history

    def evaluate(self, dataset: TokenizedData):
        results = self.model.evaluate(dataset.input,
                                      dataset.output,
                                      batch_size=64)
        return results

    def classify(self, tokenized_sentences: TokenizedData):
        results = self.model.predict(tokenized_sentences.input)[0]
        return Results(*results)

    def save(self, filename='model.h5'):
        self.model.save(filename)

    @classmethod
    def load(cls, filename='model.h5'):
        model = Model()
        model.model = load_model(filename)
        return model