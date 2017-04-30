# coding: UTF-8
import cv2
import numpy as np


def draw_boxes(img, bbox_list):
    img_drawn = np.copy(img)
    for bbox in bbox_list:
        cv2.rectangle(img_drawn, bbox[0], bbox[1], (0, 0, 255), 6)

    return img_drawn


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
