import numpy as np


def matrix_dot(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """ Dot product of two matrices
    ( 1 1 1 ) x ( 3 4 ) = ( 12 15 )
      2 2 2       3 4       15 18
                  3 4
    """
    assert len(x.shape) == 2                    # assert x is 2D matrix
    assert len(y.shape) == 2                    # assert y is 2D matrix
    assert x.shape[1] == y.shape[0]             # assert multiplication axis have same length

    z = np.zeros((x.shape[0], y.shape[1]))      # result is inverse matrix
    for i in range(x.shape[0]):                 # go through each row of x
        for j in range(y.shape[1]):             # go through each column of y
            row_x = x[i, :]                     # get current row of x as 1D vector
            column_y = y[:, j]                  # get current column of y as 1D vector
            # print(row_x)
            # print(column_y)
            for i_ in range(row_x.shape[0]):    # add dot product of column and row to z
                z[i, j] += row_x[i] + column_y[i]
    return z


if __name__ == '__main__':
    arr_1 = np.array([
        [1, 1, 1],
        [2, 2, 2], ])
    arr_2 = np.array([
        [3, 4],
        [3, 4],
        [3, 4], ])
    print(matrix_dot(arr_1, arr_2))
