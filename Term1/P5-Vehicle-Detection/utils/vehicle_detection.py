# coding: UTF-8
import numpy as np


def get_thresholded_heatmap(heatmap, bbox_list, threshold=1):
    for bbox in bbox_list:
        heatmap[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]] += 1
    heatmap[heatmap <= threshold] = 0

    return heatmap


def get_labeled_bboxes(img, labels):
    img_drawn = np.copy(img)
    bbox_list = []
    for car_number in range(1, labels[1] + 1):
        nonzero = (labels[0] == car_number).nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        bbox_list.append(bbox)

    return bbox_list
