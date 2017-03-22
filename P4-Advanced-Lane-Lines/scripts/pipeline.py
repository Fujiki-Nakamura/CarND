# coding: UTF-8
import cv2
import numpy as np

from scripts import binary_image
from scripts import camera_cal
from scripts import lane_area
from scripts import perspective_transform


class Pipeline():
    def __init__(self, line):
        self.line = line

    def process(self, img):
        # Undistort
        dist_pickle_file = './camera_cal/dist_pickle.pkl'
        mtx, dist = camera_cal.camera_calibration(dist_pickle_file)
        img_undistorted = cv2.undistort(img, mtx, dist, None, mtx)
        # Binary image
        img_binary = binary_image.process(img_undistorted, s_thresh=(170, 255), sx_thresh=(20, 100))
        # Binary and warped image
        img_warped, src, dst = perspective_transform.process(img_binary)

        nb_windows = 9
        window_height = np.int(img_warped.shape[0] / nb_windows)
        margin = 100
        minpix = 50

        if not self.line.is_detected:
            # Line finding
            histogram = np.sum(img_warped[np.int(img_warped.shape[0] / 2):, :], axis=0)

            midpoint = np.int(histogram.shape[0] / 2)
            leftx_base = np.argmax(histogram[:midpoint])
            rightx_base = np.argmax(histogram[midpoint:]) + midpoint

            nonzero = img_warped.nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])

            leftx_current = leftx_base
            rightx_current = rightx_base

            left_lane_inds = []
            right_lane_inds = []

            for window in range(nb_windows):
                win_y_low = img_warped.shape[0] - (window + 1) * window_height
                win_y_high = img_warped.shape[0] - window * window_height

                win_xleft_low = leftx_current - margin
                win_xleft_high = leftx_current + margin
                win_xright_low = rightx_current - margin
                win_xright_high = rightx_current + margin
                # Identify the nonzero pixels in x and y within the window
                good_left_inds = \
                    ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                     (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)
                    ).nonzero()[0]
                good_right_inds = \
                    ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                     (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)
                    ).nonzero()[0]

                left_lane_inds.append(good_left_inds)
                right_lane_inds.append(good_right_inds)

                if len(good_left_inds) > minpix:
                    leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
                if len(good_right_inds) > minpix:
                    rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)

            leftx = nonzerox[left_lane_inds]
            lefty = nonzeroy[left_lane_inds]
            rightx = nonzerox[right_lane_inds]
            righty = nonzeroy[right_lane_inds]
            # Fit a second order polynomial
            left_fit = np.polyfit(lefty, leftx, 2)
            right_fit = np.polyfit(righty, rightx, 2)
            # Update fits
            self.line.update(left_fit, right_fit)
        else:
            nonzero = img_warped.nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Use fits found already
            left_fit = self.line.left_fit
            right_fit = self.line.right_fit
            left_lane_inds = \
                ((nonzerox > (left_fit[0] * (nonzeroy**2)
                  + left_fit[1]*nonzeroy + left_fit[2] - margin)) &
                 (nonzerox < (left_fit[0] * (nonzeroy**2)
                  + left_fit[1] * nonzeroy + left_fit[2] + margin)))
            right_lane_inds = \
                ((nonzerox > (right_fit[0] * (nonzeroy**2)
                 + right_fit[1]*nonzeroy + right_fit[2] - margin)) &
                 (nonzerox < (right_fit[0] * (nonzeroy**2)
                  + right_fit[1]*nonzeroy + right_fit[2] + margin)))
            # Extract left and right line pixel positions
            leftx = nonzerox[left_lane_inds]
            lefty = nonzeroy[left_lane_inds]
            rightx = nonzerox[right_lane_inds]
            righty = nonzeroy[right_lane_inds]
            # Fit a second order polynomial
            left_fit = np.polyfit(lefty, leftx, 2)
            right_fit = np.polyfit(righty, rightx, 2)
            # Update fits
            self.line.update(left_fit, right_fit)

        # Generate x and y values for plotting
        ploty = np.linspace(0, img_warped.shape[0] - 1, img_warped.shape[0])
        left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

        # Draw lane area on the undistorted image
        result = lane_area.process(img_undistorted, src, dst, left_fitx, right_fitx)

        # Add information of curvature and offset from center
        ym_per_pix = 30 / 720  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 640  # meters per pixel in x dimension

        # How left from the center of the lines
        y_eval = np.max(ploty)
        loc_left = left_fit[0] * y_eval**2 + left_fit[1] * y_eval + left_fit[2]
        loc_right = right_fit[0] * y_eval**2 + right_fit[1] * y_eval + right_fit[2]
        loc_center = (loc_left + loc_right) / 2
        offset = np.round((img.shape[1] / 2 - loc_center) * (-1) * xm_per_pix, 2)

        # Curvature
        left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
        right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)

        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2 * left_fit_cr[0])
        right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2 * right_fit_cr[0])
        curvature = np.int(np.round((left_curverad + right_curverad) / 2))

        position_str = 'left' if offset >= 0 else 'right'
        text_curvature = 'Radius of Curvature = {}(m)'.format(curvature)
        text_location = 'Vehicle is {}m {} of center'.format(abs(offset), position_str)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(result, text_curvature, (25, 50), font, 1.5, (255, 255, 255), 3)
        cv2.putText(result, text_location, (25, 100), font, 1.5, (255, 255, 255), 3)

        return result
