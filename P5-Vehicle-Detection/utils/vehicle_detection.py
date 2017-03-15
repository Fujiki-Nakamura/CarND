# coding: UTF-8
import glob
import os

import cv2
import numpy as np

from skimage.feature import hog

from .classifier import get_color_hist_features


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


def draw_labeled_bboxes(img, labels):
    img_drawn = np.copy(img)

    for car_number in range(1, labels[1] + 1):
        nonzero = (labels[0] == car_number).nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        cv2.rectangle(img_drawn, bbox[0], bbox[1], (0, 0, 255), 6)

    return img_drawn


def find_cars(
    img, y_start, y_stop, classifier, X_scaler,
    orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):

    bbox_list = []
    resize_to = (64, 64)

    assert pixels_per_cell[0] == pixels_per_cell[1]
    pix_per_cell = pixels_per_cell[0]

    img_to_search = img[y_start:y_stop, :, :]
    img_to_search_ycrcb = cv2.cvtColor(img_to_search, cv2.COLOR_RGB2YCR_CB)
    subimg = cv2.cvtColor(img_to_search, cv2.COLOR_RGB2HLS)

    channel_list = []
    for i in [0, 1, 2]:
        channel_list.append(img_to_search[:, :, i])
        # channel_list.append(img_to_search_ycrcb[:, :, i])
    # Compute individual channel HOG features
    hog_ch_list = []
    for channel in channel_list:
        hog_ch = hog(
            channel, orientations, pixels_per_cell, cells_per_block,
            transform_sqrt=True, visualise=False, feature_vector=False)
        hog_ch_list.append(hog_ch)

    # Define blocks and steps as above
    n_x_blocks = (img_to_search.shape[1] // pix_per_cell) - 1
    n_y_blocks = (img_to_search.shape[0] // pix_per_cell) - 1

    window = pixels_per_cell[0] * pixels_per_cell[1]
    n_blocks_per_window = (window // pix_per_cell) - 1
    cells_per_step = 2
    n_x_steps = (n_x_blocks - n_blocks_per_window) // cells_per_step
    n_y_steps = (n_y_blocks - n_blocks_per_window) // cells_per_step

    # Window search
    for xb in range(n_x_steps):
        for yb in range(n_y_steps):
            y_pos = yb * cells_per_step
            x_pos = xb * cells_per_step
            x_left = x_pos * pix_per_cell
            y_top = y_pos * pix_per_cell

            # Extract HOG
            hog_features_list = []
            for hog_ch in hog_ch_list:
                hog_features_list.append(hog_ch[y_pos:y_pos + n_blocks_per_window, x_pos:x_pos + n_blocks_per_window].ravel())
            hog_features = np.hstack(hog_features_list).reshape(1, -1)

            patch = subimg[y_top:y_top + window, x_left:x_left + window]
            patch_resized = cv2.resize(patch, resize_to)
            # Extract image features
            # img_features = patch_resized.ravel().reshape(1, -1)
            # Extract color hist features
            color_hist_features = get_color_hist_features(patch_resized, channels=[0, 1, 2]).reshape(1, -1)
            # Scale features
            # test_features = np.array(hog_features)
            test_features = np.hstack((hog_features, color_hist_features)).reshape(1, -1)
            X_test = X_scaler.transform(test_features)
            # Predict
            y_pred_test = classifier.predict(X_test)

            if y_pred_test == 1:
                x_box_left = np.int(x_left)
                y_top_draw = np.int(y_top)
                win_draw = np.int(window)
                bbox_list.append(
                    ((x_box_left, y_top_draw + y_start),
                     (x_box_left + win_draw, y_top_draw + win_draw + y_start))
                     )

    return bbox_list
