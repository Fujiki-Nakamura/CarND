# coding: UTF-8
import glob
import os
import pickle

import cv2
import numpy as np


def camera_calibration(dist_pickle_file):
    if os.path.exists(dist_pickle_file):
        # if we have a camera matrix and dist coeffs, we use them.
        with open(dist_pickle_file, 'rb') as f:
            dist_pickle = pickle.load(f)
        mtx = dist_pickle['mtx']
        dist = dist_pickle['dist']
    else:
        # otherwise we calculate (and save) them.
        row, col = 6, 9
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ...., (6,5,0)
        objp = np.zeros((row*col, 3), np.float32)
        objp[:, :2] = np.mgrid[0:col, 0:row].T.reshape(-1, 2)
        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d points in real world space
        imgpoints = [] # 2d points in image plane.
        # Make a list of calibration images
        images = glob.glob('./camera_cal/calibration*.jpg')
        # Step through the list and search for chessboard corners
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (col, row), None)
            # If found, add object points, image points
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)

                # Draw and display the corners
                # img = cv2.drawChessboardCorners(img, (col, row), corners, ret)
                # cv2.imshow('img',img)
                # cv2.waitKey(500)
        cv2.destroyAllWindows()

        assert len(objpoints) == len(imgpoints)

        # do camera calibration
        img_size = (img.shape[1], img.shape[0])
        ret, mtx, dist, rvecs, tvecs = \
            cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
        # save the camera matrix and dist coeffs
        dist_pickle = {}
        dist_pickle['mtx'] = mtx
        dist_pickle['dist'] = dist
        with open('./camera_cal/dist_pickle.pkl', 'wb') as f:
            pickle.dump(dist_pickle, f)

    return mtx, dist


if __name__ == '__main__':
    # get the camera matrix and the dist coeffs
    mtx, dist = camera_calibration('./camera_cal/dist_pickle_file.pkl')

    test_image = './camera_cal/calibration3.jpg'
    img = cv2.imread(test_image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_undistorted = cv2.undistort(img, mtx, dist, None, mtx)
    cv2.imwrite('test_undistorted.jpg', img_undistorted)
