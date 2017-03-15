# coding: UTF-8
import glob
import os

import cv2
import numpy as np

from skimage.feature import hog


def get_hog_features(imgs, hog_channel='all', orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
    hog_features_list = []
    for img in imgs:
        if hog_channel == 'all':
            hog_features = []
            for channel in range(img.shape[2]):
                hog_feature = \
                    hog(img[:, :, channel],
                        orientations=orientations,
                        pixels_per_cell=pixels_per_cell,
                        cells_per_block=cells_per_block,
                        transform_sqrt=True,
                        visualise=False,
                        feature_vector=True)
                hog_features.append(hog_feature)
            hog_features = np.ravel(hog_features)
        else:
            hog_features = \
                hog(img[:, :, hog_channel],
                    orientations=orientations,
                    pixels_per_cell=pixels_per_cell,
                    cells_per_block=cells_per_block,
                    transform_sqrt=True,
                    visualise=False,
                    feature_vector=True)
        hog_features_list.append(hog_features)

    return np.array(hog_features_list)


def load_data(color_conversion):
    img_files_vehicle = \
        glob.glob(os.path.join('./data', 'vehicles', '*', '*.png'))
    img_files_nonvehicle = \
        glob.glob(os.path.join('./data', 'non-vehicles', '*', '*.png'))

    img_vehicles = []
    img_nonvehicles = []
    for img_file in img_files_vehicle:
        img_vehicles.append(cv2.cvtColor(cv2.imread(img_file), color_conversion))
    for img_file in img_files_nonvehicle:
        img_nonvehicles.append(cv2.cvtColor(cv2.imread(img_file), color_conversion))

    label_vehicles = np.ones(len(img_vehicles))
    label_nonvehicles = np.zeros(len(img_nonvehicles))
    labels = np.concatenate((label_vehicles, label_nonvehicles))
    assert len(labels) == len(label_vehicles) + len(label_nonvehicles)

    imgs = np.concatenate((img_vehicles, img_nonvehicles))
    assert len(imgs) == len(img_vehicles) + len(img_nonvehicles)

    print('vehicle = {} samples'.format(len(img_vehicles)))
    print('non-vehicle = {} samples'.format(len(img_nonvehicles)))

    return imgs, labels
