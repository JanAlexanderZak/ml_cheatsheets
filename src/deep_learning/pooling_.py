import numpy as np


def find_maximum(array: np.array) -> np.ndarray:
    assert len(x.shape) == 1
    _max = array[0]
    for i in array:
        if i > _max:
            _max = i
    return [_max]


def calc_average(array: np.ndarray) -> np.ndarray:
    assert len(x.shape) == 1
    _sum = np.zeros(1)
    n = len(array)
    for i in array:
        _sum += i
    return _sum / n


def calc_l2norm(array: np.ndarray) -> np.ndarray:
    assert len(x.shape) == 1
    _sum = np.zeros(1)
    for i in array:
        _sum += i ** 2
    return _sum ** (1 / 2)


def average_pooling_1d(array, window=2):
    assert len(array.shape) == 1
    pooling_array = np.array([])

    for i in range(array.shape[0] - 1):
        pooling_array = np.append(pooling_array, calc_average(array[i:i + window]))
    return pooling_array


def maximum_pooling_1d(array, window=2) -> np.ndarray:
    assert len(array.shape) == 1
    pooling_array = np.array([])

    for i in range(array.shape[0] - 1):
        pooling_array = np.append(pooling_array, find_maximum(array[i:i + window]))
    return pooling_array


def l2norm_pooling_1d(array, window=2) -> np.ndarray:
    assert len(array.shape) == 1
    pooling_arary = np.array([])

    for i in range(array.shape[0] - 1):
        pooling_arary = np.append(pooling_arary, calc_l2norm(array[i:i + window]))
    return pooling_arary


def pooling_1d(array: np.ndarray, window=2, kind: str = 'max') -> np.ndarray:
    """ Convolution                     Pooling with:
    1. Find vector dimension            - max
    2. Even convolution operation       - avg
    3. Window slide                     - x^2
    """
    assert len(array.shape) == 1
    pooling_array = np.array([])
    array_length = array.shape[0] - 1

    if kind == 'max':
        for i in range(array_length):
            pooling_array = np.append(pooling_array, find_maximum(array[i:i + window]))
        return pooling_array

    if kind == 'avg':
        for i in range(array_length):
            pooling_array = np.append(pooling_array, calc_average(array[i:i + window]))
        return pooling_array

    if kind == 'l2':
        for i in range(array_length):
            pooling_array = np.append(pooling_array, calc_l2norm(array[i:i + window]))
        return pooling_array
    else:
        print("This 'kind' is not supported!")


if __name__ == '__main__':
    x = np.array([1, 2, 3, 4, 5])
    print("Maximum: ", find_maximum(x))
    print("Average: ", calc_average(x))
    print("L2-norm: ", calc_l2norm(x))

    print("Maximum pooling: ", maximum_pooling_1d(x))
    print("Average pooling: ", average_pooling_1d(x))
    print("L2-norm pooling: ", l2norm_pooling_1d(x))
    print(pooling_1d(x))


"""
x = tf.constant([1., 2., 3., 4., 5.])
x = tf.reshape(x, [1, 5, 1])

max_pool_1d = tf.keras.layers.MaxPooling1D(pool_size=2, strides=1, padding='valid')
print(max_pool_1d(x))
print(x.shape, max_pool_1d(x).shape)
"""
