import numpy as np


def dummy_probability_time_series(change_index: int, n: int = 100):
    x = np.arange(n)
    if change_index is None:
        y = np.random.rand(n) + np.random.choice([0, 1])
    else:
        yl = np.random.rand(change_index)
        yr = np.random.rand(n - change_index) + 100
        y = np.concatenate((yl, yr), axis=0) / 100
    return x.astype(np.float64), y.astype(np.float64)