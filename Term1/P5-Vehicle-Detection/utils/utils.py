# coding: UTF-8
import glob
import os

import cv2
import numpy as np


def draw_multi_scale_windows(img, ystart, ystop, scale, orient=8, pix_per_cell=8, cell_per_block=2):
    draw_img = np.copy(img)
    img = img.astype(np.float32) / 255

    img_tosearch = img[ystart:ystop, :, :]
    imshape = img_tosearch.shape
    img_tosearch = cv2.resize(img_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    # Define blocks and steps as above
    nxblocks = (img_tosearch.shape[1] // pix_per_cell) - 1
    nyblocks = (img_tosearch.shape[0] // pix_per_cell) - 1
    #nfeat_per_block = orient * cell_per_block ** 2

    window = 64
    nblocks_per_window = (window // pix_per_cell) - 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    #rect_start = None
    #ect_end = None
    for i, xb in enumerate(range(nxsteps+1)):
        for j, yb in enumerate(range(nysteps+1)):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            xbox_left = np.int(xleft * scale)
            ytop_draw = np.int(ytop * scale)
            win_draw = np.int(window * scale)
            rect_start = (xbox_left, ytop_draw + ystart)
            rect_end = (xbox_left + win_draw, ytop_draw + win_draw + ystart)
            cv2.rectangle(draw_img, rect_start, rect_end, (0, 0, 255), 6)
    cv2.rectangle(draw_img, rect_start, rect_end, (255, 0, 0), 6)

    return draw_img


def draw_sliding_windows(
        img,
        y_start, y_stop, scale,
        pixels_per_cell=(8, 8), cells_per_block=(2, 2)
        ):

    assert pixels_per_cell[0] == pixels_per_cell[1]
    pix_per_cell = pixels_per_cell[0]

    img_draw = np.copy(img)
    img = img.astype(np.float32) / 255

    img_to_search = img[y_start:y_stop, :, :]
    resize_to = (np.int(img_to_search.shape[1] / scale), np.int(img_to_search.shape[0] / scale))
    img_to_search = cv2.resize(img_to_search, resize_to)

    n_x_blocks = (img_to_search.shape[1] // pix_per_cell) - 1
    n_y_blocks = (img_to_search.shape[0] // pix_per_cell) - 1

    window = 64
    n_blocks_per_window = (window // pix_per_cell) - 1
    cells_per_step = 2
    n_x_steps = (n_x_blocks - n_blocks_per_window) // cells_per_step
    n_y_steps = (n_y_blocks - n_blocks_per_window) // cells_per_step

    for xb in range(n_x_steps + 1):
        for yb in range(n_y_steps + 1):
            y_pos = yb * cells_per_step
            x_pos = xb * cells_per_step
            x_left = x_pos * pix_per_cell
            y_top = y_pos * pix_per_cell

            x_box_left = np.int(x_left * scale)
            y_top_draw = np.int(y_top * scale)
            win_draw = np.int(window * scale)
            win_start = (x_box_left, y_top_draw + y_start)
            win_end = (x_box_left + win_draw, y_top_draw + win_draw + y_start)
            cv2.rectangle(img_draw, win_start, win_end, (0, 0, 255), 6)
    cv2.rectangle(img_draw, win_start, win_end, (255, 0, 0), 6)

    return img_draw


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
