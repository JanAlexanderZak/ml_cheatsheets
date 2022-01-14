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


def show_arrays(func):
    @functools.wraps(func)
    def _show_arrays(*args, **kwargs):

        value = func(*args, **kwargs)

        print(f"Matrix shape: {args[0].shape} with \n{args[0]}\n"
              f"Vector shape: {args[1].shape} with \n{args[1]}")
        print("#" * 80)
        return value
    return _show_arrays


@show_arrays
def add_loop(_x: np.ndarray, _y: np.ndarray) -> np.ndarray:
    """ Simply adds two matrices elementwise:
    . . . 32     . . . 16
    .         +  .
    .            .
    64           32

    """
    assert len(_x.shape) == 2               # assert 2D matrix
    assert len(_y.shape) == 1               # assert 1D vector
    assert _x.shape[1] == _y.shape[0]       # assert addition axis have same length

    for i in range(_x.shape[0]):            # go through each row
        for j in range(_x.shape[1]):        # and each column
            _x[i, j] += _y[j]               # add value in row of y to matrix element
    return _x


if __name__ == '__main__':
    x = np.ones((10, 2))
    y = np.ones(2)
    print(add_loop(x, y))
