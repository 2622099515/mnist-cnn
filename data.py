import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils

def load_mnist():
    img_rows = img_cols = 28

    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)

    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    return (x_train, y_train, x_test, y_test)
