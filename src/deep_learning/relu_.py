import functools
import time
import numpy as np


def timer(func):
    @functools.wraps(func)
    def _timer(*args, **kwargs):
        tic = time.perf_counter()
        value = func(*args, **kwargs)
        toc = time.perf_counter()
        print(f"Elapsed time: {toc - tic:.8f} seconds")
        return value
    return _timer


def relu_loop(x):
    """ ReLu
    out = reLu(dot(W, input) + b)
    relu = max(x, 0)
    dot product:
                     b1
    ( a1 a2 a3 ) x ( b2 ) = a1b1 + a2b2 + a3b3
                     b3
    """

    assert len(x.shape) == 2            # assert it is 2D matrix

    x = x.copy()                        # deep copy
    for i in range(x.shape[0]):         # go through each row
        for j in range(x.shape[1]):     # and each column
            x[i, j] = max(x[i, j], 0.)  # replace each value with > 0, implementation of max in ./max_pooling.py
    return x


if __name__ == '__main__':
    arr = np.array([[-1, 2, -5],
                    [1, 2, 3], ])
    print(relu_loop(arr))
