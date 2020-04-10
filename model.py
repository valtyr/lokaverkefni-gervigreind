from tokenizer import TokenizedData
from helpers import pairs
from keras.layers import Dense, Activation, Embedding, GRU, Conv1D, Dropout, Bidirectional, SpatialDropout1D, Input, Lambda, MaxPooling1D, GlobalAveragePooling1D, concatenate, LSTM
from keras.models import Model as KerasModel, Sequential
from keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer, text_to_word_sequence

import tensorflow as tf
tf.config.experimental.set_visible_devices([], 'GPU')


class Model:
    def __init__(self):
        self.model = Sequential()

    def build(self, dataset: TokenizedData):
        input_shape = dataset.input.shape
        n_comments, width = input_shape

        layers = [
            Embedding(input_dim=len(dataset.tokens) + 1,
                      output_dim=512,
                      input_length=width),
            Dropout(0.2),
            Conv1D(64, 5, padding='valid', activation='relu', strides=1),
            SpatialDropout1D(0.2),
            MaxPooling1D(pool_size=4),
            LSTM(70),
            Dense(32),
            Dense(6),
            Activation('softmax')
        ]
        #input, embedding, spatialdropuot1d globalmaxpooling, concatenate, dropout, dense ,dense

        for layer in layers:
            self.model.add(layer)

        optimizer = Adam(learning_rate=0.01)
        self.model.compile(optimizer=optimizer,
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def fit(self, dataset: TokenizedData):
        history = self.model.fit(dataset.input,
                                 dataset.output,
                                 batch_size=64,
                                 verbose=True,
                                 epochs=3,
                                 validation_split=0.4)
        return history

    def evaluate(self, dataset: TokenizedData):
        results = self.model.evaluate(dataset.input,
                                      dataset.output,
                                      batch_size=64)
        return results