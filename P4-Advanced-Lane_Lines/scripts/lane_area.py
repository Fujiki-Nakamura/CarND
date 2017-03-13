# coding: UTF-8
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


def process(img_undistorted, src, dst, left_fitx, right_fitx):
    # Create an image to draw the lines on
    color_warp = np.zeros_like(img_undistorted).astype(np.uint8)

    # Recast the x and y points into usable format for cv2.fillPoly()
    ploty = np.linspace(0, img_undistorted.shape[0] - 1, img_undistorted.shape[0])
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    Minv = cv2.getPerspectiveTransform(dst, src)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img_undistorted.shape[1], img_undistorted.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(img_undistorted, 1, newwarp, 0.3, 0)

    return result


if __name__ == '__main__':
    pass
