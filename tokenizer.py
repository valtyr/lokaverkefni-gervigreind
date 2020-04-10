from collections import namedtuple
from data import Dataset
from helpers import pad

import numpy as np

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
            padded_input = pad(sequences, self.width)

        return TokenizedData(padded_input, self.word_index, data.output,
                             data.input)
