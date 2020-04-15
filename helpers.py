import numpy as np


def pairs(items: list):
    for i in range(1, len(items)):
        yield (items[i], items[i - 1])


def pad(sequences: list, width: int):
    padded_list = [list(row) + [0] * (width - len(row)) for row in sequences]
    import ipdb
    ipdb.set_trace()
    return np.array(padded_list)
