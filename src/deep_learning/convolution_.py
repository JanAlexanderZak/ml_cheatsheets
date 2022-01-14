import numpy as np


def convolution_1d(array: np.ndarray, _kernel: np.ndarray) -> np.ndarray:
    """ Convolution
    1. Find vector dimension
    2. Even convolution operation
    3. Window slide
    """
    assert len(array.shape) == 1
    convolution = np.zeros(array.shape[0])
    kernel_length = _kernel.shape[0]
    # print(f"Input dimension: {array.shape[0]} \nOutput dimension: {array.shape[0] - kernel_length + 1}")

    for i in range(array.shape[0] - kernel_length + 1):
        convolution[i] = np.dot(array[i:i + kernel_length], _kernel)
    return convolution


if __name__ == '__main__':
    x = np.array([1, 2, 3, 4, 5])
    kernel = np.array([0, 0, 1])
    print(convolution_1d(x, kernel))
