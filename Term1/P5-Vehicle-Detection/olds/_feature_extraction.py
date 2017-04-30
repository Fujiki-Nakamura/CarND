# coding: UTF-8
import glob
import os

import cv2
import numpy as np

from skimage.feature import hog


def bin_spatial(img, size=(32, 32)):
    color1 = cv2.resize(img[:, :, 0], size).ravel()
    color2 = cv2.resize(img[:, :, 1], size).ravel()
    color3 = cv2.resize(img[:, :, 2], size).ravel()

    return np.hstack((color1, color2, color3))


def color_hist(img, channels=[0, 1, 2], n_bins=32):
    color_hist_features_list = []
    for channel in channels:
        color_hist_features_list.append(np.histogram(img[:, :, channel], bins=n_bins)[0])

    return np.concatenate(color_hist_features_list)


def get_hog_features(
        img, hog_channels=[0, 1, 2],
        orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
        feature_vector=True):
    if hog_channels is None:
        return hog(
                img,
                orientations=orientations,
                pixels_per_cell=pixels_per_cell,
                cells_per_block=cells_per_block,
                transform_sqrt=True,
                visualise=False,
                feature_vector=feature_vector)

    hog_features = []
    for channel in hog_channels:
        hog_feature = \
            hog(img[:, :, channel],
                orientations=orientations,
                pixels_per_cell=pixels_per_cell,
                cells_per_block=cells_per_block,
                transform_sqrt=True,
                visualise=False,
                feature_vector=feature_vector)
        hog_features.append(hog_feature)

    return np.hstack(hog_features)


def draw_boxes(img, bbox_list):
    img_drawn = np.copy(img)
    for bbox in bbox_list:
        cv2.rectangle(img_drawn, bbox[0], bbox[1], (0, 0, 255), 6)

    return img_drawn


def add_heat(heatmap, bbox_list):
    for bbox in bbox_list:
        heatmap[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]] += 1

    return heatmap


def apply_threshold(heatmap, threshold):
    heatmap[heatmap <= threshold] = 0

    return heatmap


def get_labeled_bboxes(img, labels):
    img_drawn = np.copy(img)
    bboxes = []

    for car_number in range(1, labels[1] + 1):
        nonzero = (labels[0] == car_number).nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        bboxes.append(((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy))))

    return bboxes


def find_cars(
    img,
    convert_to,
    y_start, y_stop, scale,
    classifier, X_scaler,
    target_channel=[0, 1, 2],
    orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):

    assert pixels_per_cell[0] == pixels_per_cell[1]
    pix_per_cell = pixels_per_cell[0]

    img = img.astype(np.float32) / 255
    img_to_search = img[y_start:y_stop, :, :]
    img_to_search_converted = cv2.cvtColor(img_to_search, convert_to)

    if scale != 1:
        resize_to = (
            np.int(img_to_search_converted.shape[1] / scale),
            np.int(img_to_search_converted.shape[0] / scale)
            )
        img_to_search_converted = cv2.resize(img_to_search_converted, resize_to)

    # Compute individual channel HOG features
    hog_ch_list = []
    for channel in target_channel:
        hog_ch = hog(
            img_to_search_converted[:, :, channel],
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            transform_sqrt=True,
            visualise=False,
            feature_vector=False)
        hog_ch_list.append(hog_ch)

    # Define blocks and steps as above
    n_x_blocks = (img_to_search_converted.shape[1] // pix_per_cell) - 1
    n_y_blocks = (img_to_search_converted.shape[0] // pix_per_cell) - 1

    window = pixels_per_cell[0] * pixels_per_cell[1]
    n_blocks_per_window = (window // pix_per_cell) - 1
    cells_per_step = 2
    n_x_steps = (n_x_blocks - n_blocks_per_window) // cells_per_step
    n_y_steps = (n_y_blocks - n_blocks_per_window) // cells_per_step

    # Window search
    bbox_list = []
    draw_img = np.copy(img)
    for xb in range(n_x_steps):
        for yb in range(n_y_steps):
            y_pos = yb * cells_per_step
            x_pos = xb * cells_per_step
            x_left = x_pos * pix_per_cell
            y_top = y_pos * pix_per_cell

            # Extract HOG features
            hog_features_list = []
            for hog_ch in hog_ch_list:
                hog_features_list.append(hog_ch[y_pos:y_pos + n_blocks_per_window, x_pos:x_pos + n_blocks_per_window].ravel())
            hog_features = np.hstack(hog_features_list)

            subimg = img_to_search_converted[y_top:y_top + window, x_left:x_left + window]
            subimg_resized = cv2.resize(subimg, (64, 64))

            # Extract image features
            img_features = bin_spatial(subimg_resized)
            # Extract color hist features
            color_hist_features = get_color_hist_features(subimg_resized, channels=[0, 1, 2])

            test_features = np.hstack((hog_features, color_hist_features, img_features))
            # Scale features
            X_test = X_scaler.transform(test_features).reshape(1, -1)
            # Predict
            y_pred_test = classifier.predict(X_test)

            if y_pred_test == 1:
                x_box_left = np.int(x_left * scale)
                y_top_draw = np.int(y_top * scale)
                win_draw = np.int(window * scale)
                bbox_list.append(
                    ((x_box_left, y_top_draw + y_start),
                     (x_box_left + win_draw, y_top_draw + win_draw + y_start))
                     )

    return bbox_list
