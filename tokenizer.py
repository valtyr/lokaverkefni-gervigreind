from collections import namedtuple
from data import Dataset
from helpers import pad

import numpy as np
import pickle

from keras.preprocessing.text import Tokenizer as KerasTokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences

TokenizedData = namedtuple('TokenizedData',
                           ('input', 'tokens', 'output', 'original_input'))


class Tokenizer:
    width = None

    def __init__(self):
        self.tokenizer = KerasTokenizer()

    def fit(self, data: Dataset):
        self.tokenizer.fit_on_texts(data.input)
        self.word_index = self.tokenizer.word_index

    def tokenize(self, data: Dataset):
        sequences = self.tokenizer.texts_to_sequences(data.input)
        if self.width is None:
            padded_input = pad_sequences(sequences, padding='post')
            self.width = padded_input.shape[1]
        else:
            padded_input = pad_sequences(sequences, self.width)

        return TokenizedData(padded_input, self.word_index, data.output,
                             data.input)

    def tokenize_sentence(self, sentence: str):
        sequences = self.tokenizer.texts_to_sequences([sentence])
        padded_input = pad_sequences(sequences, self.width)

        return TokenizedData(padded_input, self.word_index, None, [sentence])

    def save(self, filename='tokenizer.pickle'):
        with open('tokenizer.pickle', 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, filename='tokenizer.pickle'):
        with open('tokenizer.pickle', 'rb') as handle:
            return pickle.load(handle)