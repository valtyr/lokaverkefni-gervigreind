import pandas as pd
import numpy as np

from collections import namedtuple

Dataset = namedtuple('Dataset', ('input', 'output'))
column_labels = [
    'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'
]


def unique_words(df):
    df['comment_text'].str.lower().str.split()
    results = set()
    df['comment_text'].str.lower().str.split().apply(results.update)
    return len(results)


def load_data(filename='train.csv', sample_size=10000, data_split=0.4):
    data = pd.read_csv(f'./data/{filename}')

    data['none'] = 1 - data[column_labels].max(axis=1)
    data['comment_text'].fillna("unknown", inplace=True)

    n_train = int(sample_size * data_split)
    n_validation = sample_size - n_train

    training_data = data[:n_train]
    validation_data = data[n_train:n_train + n_validation]

    return (Dataset(training_data['comment_text'],
                    training_data[column_labels]),
            Dataset(validation_data['comment_text'],
                    validation_data[column_labels]))
