""" Define a Neural Network model like NVIDIA model.
    The original model is described in https://arxiv.org/pdf/1604.07316v1.pdf
"""
from keras.layers import Convolution2D
from keras.layers import Cropping2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Lambda
from keras.models import Sequential


def get_model():
    input_shape = (160, 320, 3)

    model = Sequential()
    # Cropping layer
    model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=input_shape))
    # Normalizing layer
    model.add(Lambda(lambda x: x / 255 - 0.5))

    model.add(
        Convolution2D(
            24, 5, 5,
            border_mode='valid', subsample=(2, 2),
            # W_regularizer=l2(0.001),
            activation='relu'
            )
        )

    model.add(
        Convolution2D(
            36, 5, 5,
            border_mode='valid',
            subsample=(2, 2),
            # W_regularizer=l2(0.001),
            activation='relu'
            )
        )

    model.add(
        Convolution2D(
            48, 5, 5,
            border_mode='valid',
            subsample=(2, 2),
            # W_regularizer=l2(0.001),
            activation='relu'
            )
        )

    model.add(
        Convolution2D(
            64, 3, 3,
            border_mode='valid',
            subsample=(2, 2),
            # W_regularizer = l2(0.001)
            activation='relu'
            )
        )

    model.add(
        Convolution2D(
            64, 3, 3,
            border_mode='valid',
            subsample=(2, 2),
            # W_regularizer=l2(0.001)
            activation='relu'
            )
        )

    model.add(Flatten())

    model.add(Dense(64))
    model.add(Dropout(0.5))

    model.add(Dense(32))
    model.add(Dropout(0.5))

    model.add(Dense(16))
    model.add(Dropout(0.5))

    model.add(Dense(8))

    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')

    return model
