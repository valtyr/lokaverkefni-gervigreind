import pandas as pd
import numpy as np

from collections import namedtuple

Dataset = namedtuple('Dataset', ('input', 'output'))
column_labels = [
    'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'
]

ResultsTuple = namedtuple('Results', column_labels)


class Results(ResultsTuple):
    def __repr__(self):
        values = list(self)
        max_idx = np.argmax(values)
        return f'<Results ({column_labels[max_idx]}, certainty: {values[max_idx]})>'


def unique_words(df):
    df['comment_text'].str.lower().str.split()
    results = set()
    df['comment_text'].str.lower().str.split().apply(results.update)
    return len(results)


def load_data(filename='train.csv', sample_size=None, data_split=0.4):
    data = pd.read_csv(f'./data/{filename}')

    data['none'] = 1 - data[column_labels].max(axis=1)
    data['comment_text'].fillna("unknown", inplace=True)

    if sample_size is None:
        sample_size = data.shape[0]

    n_train = int(sample_size * data_split)
    n_validation = sample_size - n_train

    training_data = data[:n_train]
    validation_data = data[n_train:n_train + n_validation]

    return (Dataset(training_data['comment_text'],
                    training_data[column_labels]),
            Dataset(validation_data['comment_text'],
                    validation_data[column_labels]))
