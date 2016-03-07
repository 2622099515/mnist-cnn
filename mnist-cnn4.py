import numpy as np
from keras.models import Sequential
from keras.layers.core import Dropout, Flatten, Dense, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from data import load_mnist

def mnist_cnn(x_train, y_train, x_test, y_test):
    batch_size = 128
    nb_epoch = 16

    model = Sequential()

    model.add(Convolution2D(4, 5, 5, border_mode='valid',
                            input_shape=(1, 28, 28)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(12, 5, 5, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(10))
    model.add(Activation('softmax'))

    sgd = SGD(lr=0.01, momentum=0.9, decay=1e-6, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')

    model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,
            show_accuracy=True, verbose=1, shuffle=True,
            validation_data=(x_test, y_test))

    score = model.evaluate(x_test, y_test, show_accuracy=True, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

X_train, Y_train, X_test, Y_test = load_mnist()
mnist_cnn(X_train, Y_train, X_test, Y_test)
