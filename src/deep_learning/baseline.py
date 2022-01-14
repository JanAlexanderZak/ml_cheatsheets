import tensorflow as tf
import numpy as np
import copy
from tensorflow import keras


def calculate_baseline(_targets: list) -> float:
    """ Probability of randomly guessing the correct target
    Precondition: All classes are evenly distributed!
    :param _targets: list of all possible targets
    :return: baseline probability
    """
    test_data = np.repeat(_targets, 10000)
    test_data_copy = copy.copy(test_data)
    np.random.shuffle(test_data_copy)

    hits_array = np.array(test_data) == np.array(test_data_copy)
    return np.sum(hits_array) / len(test_data)


def naive_prediction(_y_true, _y_pred):
    # Computes the mean of squares of errors between labels and predictions
    return np.mean(keras.losses.mean_squared_error(_y_true, _y_pred))


def linear_prediction(_x_train, _y_train, _x_valid, _y_valid):
    # Computes baseline with linear nn
    model = keras.models.Sequential([
        keras.layers.Flatten(),
        keras.layers.Dense(1), ])
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.1), loss='mean_absolute_error', metrics=['acc'])
    model.fit(_x_train, _y_train, epochs=30, validation_data=(_x_valid, _y_valid))
    result = model.evaluate(_x_valid, _y_valid)
    return result


if __name__ == '__main__':
    # Manual
    targets = [3, 2, 1]
    print(calculate_baseline(targets))

    # Naive
    y_true = [[0., 1.], [0., 0.]]
    y_pred = [[1., 1.], [1., 0.]]
    print(naive_prediction(y_true, y_pred))

    # Linear
    x_train = [-1, 2, -3, 4, -5, 6, -7, 8, -9, 10]
    y_train = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    x_val = [-11, 12, -13, 14, -15, 16, -17, 18, -19, 20]
    y_val = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    print(linear_prediction(x_train, y_train, x_val, y_val))

