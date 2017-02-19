import gc
import time

import cv2

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

import keras.backend as K
from keras.callbacks import EarlyStopping

import model_nvidia_like


# path to the data directory
data_dir = './data/'
# Load the driving log data
df_drive_log = pd.read_csv(data_dir + 'driving_log.csv')


def generator(df_drive_log, batch_size=128):
    nb_samples = df_drive_log.shape[0]
    while True:
        sklearn.utils.shuffle(df_drive_log)

        for offset in range(0, nb_samples, batch_size):
            df_drive_log_batches = df_drive_log[offset:offset + batch_size]

            images = []
            steerings = []

            for index, row in df_drive_log_batches.iterrows():
                correction = 0.2
                # center image
                img_file_center = data_dir + row['center'].strip()
                img_center = cv2.imread(img_file_center)
                img_center = cv2.cvtColor(img_center, cv2.COLOR_BGR2RGB)
                # left image
                img_file_left = data_dir + row['left'].strip()
                img_left = cv2.imread(img_file_left)
                img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB)
                # right image
                img_file_right = data_dir + row['right'].strip()
                img_right = cv2.imread(img_file_right)
                img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB)
                # steering
                steering_center = row['steering']
                steering_left = steering_center + correction
                steering_right = steering_center - correction

                images.extend([img_center, img_left, img_right])
                steerings.extend([steering_center, steering_left, steering_right])

            X = np.array(images)
            y = np.array(steerings)

            yield sklearn.utils.shuffle(X, y)


def flip(image):
    """horizontal flip the image"""
    image_flipped = cv2.flip(image, 1)
    angle_flipped = -1 * angle
    return image_flipped, angle_flipped


def main():
    batch_size = 128

    # Prepare the data
    # train val split
    df_drive_log_train, df_drive_log_val = \
        train_test_split(df_drive_log, test_size=0.25, random_state=0)

    # create the model (the model is defined in another script)
    model = model_nvidia_like.get_model()
    # generators
    train_generator = generator(df_drive_log_train, batch_size=batch_size)
    val_generator = generator(df_drive_log_val, batch_size=batch_size)
    # callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    # Train the model
    model.fit_generator(
        generator=train_generator,
        samples_per_epoch=df_drive_log_train.shape[0],
        nb_epoch=10,
        validation_data=val_generator,
        nb_val_samples=df_drive_log_val.shape[0]
    )

    # Save the model
    model.save('model.h5')
    print('The model saved.')

    K.clear_session()
    gc.collect()


if __name__ == '__main__':
    main()
