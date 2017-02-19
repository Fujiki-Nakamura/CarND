import gc
import time

import cv2

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import keras.backend as K
from keras.callbacks import EarlyStopping

import model_nvidia_like


def load_data():
    # TODO: use the left and right images
    df_drive_log = pd.read_csv('./data/driving_log.csv')
    y = df_drive_log['steering'].values
    X = []
    images = df_drive_log['center'].values
    for img in images:
        img_file = './data/' + img
        img = cv2.imread(img_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        X.append(img)
    X = np.array(X)

    return X, y


def flip(image):
    """horizontal flip the image"""
    image_flipped = cv2.flip(image, 1)
    angle_flipped = -1 * angle
    return image_flipped, angle_flipped


def main():
    # Prepare the data
    t0 = time.time()
    X, y = load_data()
    print('Loading time = {} seconds'.format(time.time() - t0))
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.25, random_state=0)

    # Train the model
    batch_size = 128
    model = model_nvidia_like.get_model()
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
    print('The model saved.')

    K.clear_session()
    gc.collect()


if __name__ == '__main__':
    main()
