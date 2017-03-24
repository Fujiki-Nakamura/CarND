import time

import cv2

import pandas as pd
from sklearn.model_selection import train_test_split

from keras.callbacks import EarlyStopping
from keras.layers import Dense, Activation
from keras.layers import Convolution2D
from keras.layers import Cropping2D
from keras.layers import Dropout
from keras.layers import GlobalAveragePooling2D
from keras.layers import Input
from keras.layers import Lambda
from keras.layers import MaxPooling2D
from keras.layers import merge


def fire_module(previous_layer, fire_id, squeeze_unit=16, expand_unit=64):
    fire_squeeze = \
        Convolution2D(
            16, 1, 1,
            init='glorot_uniform',
            activation='relu',
            border_mode='same',
            name='fire{}_squeeze'.format(fire_id)
        )(previous_layer)

    fire_expand1 = \
        Convolution2D(
            64, 1, 1,
            init='glorot_uniform',
            activation='relu',
            border_mode='same',
            name='fire{}_expand1'
        )(fire_squeeze)

    fire_expand2 = \
        Convolution2D(
            64, 3, 3,
            init='glorot_uniform',
            activation='relu',
            border_mode='same',
            name='fire{}_expand2'
        )(fire_squeeze)

    merge = \
        merge(
            [fire_expand1, fire_expand2],
            mode='concat',
            concat_axis=3,
            name='fire{}_merge'.format(fire_id)
        )

    return merge


def get_model():
    input_shape = (160, 320, 3)

    input_img = Input(shape=input_shape)
    # Croppping layer
    img_cropped = \
    Cropping2D(cropping=((50, 20), (0, 0)), input_shape=input_shape)(input_img)
    # Normalizing layer
    img_normalized = \
        Lambda(lambda x: x / 127.5 - 1, input_shape=input_shape)(img_cropped)

    conv1 = \
        Convolution2D(
            96, 7, 7,
            init='glorot_uniform',
            activation='relu',
            border_mode='same',
            subsample=(2, 2),
            name='conv1')(img_normalized)
    maxpool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='maxpool1')(conv1)

    fire_module2 = fire_module(conv1, fire_id=2, squeeze_unit=16, expand_unit=64)
    fire_module3 = fire_module(fire_module2, fire_id=3, squeeze_unit=16, expand_unit=64)
    fire_module4 = fire_module(fire_module3, fire_id=4, squeeze_unit=32, expand_unit=128)
    maxpool4 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='maxpool2')(fire_module4)

    fire_module5 = fire_module(maxpool4, squeeze_unit=32, expand_unit=128)
    fire_module6 = fire_module(fire_module5, squeeze_unit=48, expand_unit=192)
    fire_module7 = fire_module(fire_module6, squeeze_unit=48, expand_unit=192)
    fire_module8 = fire_module(fire_module7, squeeze_unit=64, expand_unit=256)
    maxpool8 = MaxPooling(pool_size=(3, 3), strides=(2, 2), name='maxpool8')(fire_module8)

    fire_module9 = fire_module(maxpool8, squeeze_unit=64, expand_unit=256)
    dropout9 = Dropout(0.5, name='dropout9')(fire_module9)

    # conv10 = \
    #     Convolution2D(
    #         1000, 1, 1,
    #         init='glorot_uniform',
    #         border_mode='valid',
    #         subsample=(2, 2),
    #         name='conv10'
    #     )(dropout9)

    global_avgpool9 = GlobalAveragePooling2D()(dropout9)
    output = Dense(1, name='output')(global_avgpool9)

    model = Model(input=input_img, output=output)
    model.compile(loss='mse', optimizer='adam')

    return model


def load_data():
    df_drive_log = pd.read_csv('./data/driving_log.csv')
    y = df_drive_log['steering'].values
    X = []
    images = df_drive_log['center'].values
    for img in images:
        img_file = './data/' + img
        img = cv2.imread(img_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        X.append(img)

    return X, y


def main():
    # Prepare the data
    t0 = time.time()
    X, y = load_data()
    print('Loading time = {} seconds'.format(time.time() - t0))
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.25, random_state=0)

    # Train the model
    batch_size = 128
    model = get_model()
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    model.fit(
        X_train, y_train,
        batch_size=batch_size, nb_epoch=10,
        callbacks=[early_stopping],
        validation_split=0.25,
        )

    # Test the model
    model.evaluate(X_test, y_test, batch_size=batch_size)

    # Save the model
    model.save('model.h5')


if __name__ == '__main__':
    main()
