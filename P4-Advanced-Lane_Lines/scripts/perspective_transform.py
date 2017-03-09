# coding: UTF-8
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

def pipeline(img):
    img_size = img.shape[1::-1]
    # source points
    src = np.float32(
        [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
        [((img_size[0] / 6) - 10), img_size[1]],
        [(img_size[0] * 5 / 6) + 60, img_size[1]],
        [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
    # destination points
    dst = np.float32(
        [[(img_size[0] / 4), 0],
        [(img_size[0] / 4), img_size[1]],
        [(img_size[0] * 3 / 4), img_size[1]],
        [(img_size[0] * 3 / 4), 0]])

    M = cv2.getPerspectiveTransform(src, dst)
    img_warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

    return img_warped


if __name__ == '__main__':
    pass
