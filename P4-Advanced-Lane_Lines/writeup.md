## Writeup of Advanced Lane Finding Project

[//]: # (Image References)

[undistort1]: ./output_images/undistorted/calibration3_compare.jpg "Undistorted"
[undistort2]: ./output_images/undistorted/test1_compare.jpg "Undistorted"
[undistort3]: ./output_images/undistorted/test5.jpg "Undistorted"
[binary1]: ./output_images/binary_image.png "Binary Example"
[warp]: ./output_images/perspective_transform.png "Warp Example"
[histogram]: ./output_images/histogram.png "Histogram"
[lane]: ./output_images/lane_detection.png "Fit Visual"
[output]: ./output_images/pipeline.png "Output"

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.


### Camera Calibration
First, I prepared "object points", which will be the (x, y, z) coordinates of the chessboard corners in in real world space. Then, using `cv2.findChessboardCorners()` function, I got the chessboard corners in the image as "image points". With object points and image points, I calculated the camera calibration and obtained a camera matrix and distortion coefficients using `cv2.calibrateCamera()` function. Finally, I applied this distortion correction to a test image using the `cv2.undistort()` function and obtained the result below.
<br>The codes used in this step are in `scripts/camera_cal.py`
![alt text][undistort1]

### Image Undistortion
Using the camera matrix and distortion coefficients I got in the camera calibration step and `cv2.undistort()` function, I undistorted the raw image of driving.
<br>Following is an example of an undistorted image:
![alt text][undistort2]

### Image Thresholding
I applied a combination of color and gradient thresholds to the test image and generated a binary image. The color thresholding was applied to the S channel of the image converted from RGB to HLS.
<br>The codes used in this step are in `scripts/binary_image.py`
<br>Following is an example of a binary image:
![alt text][binary1]

### Perspective Transformation
The perspective transformation was done with the source points and destination points defined as below. The codes for this step are in `scripts/perspective_transform.py`.

```python
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
```

This resulted in the following source and destination points:

| source | destination |
| --- | --- |
| (585, 460) | (320, 0) |
| (203.33, 720) | (320, 720) |
| (1226.67, 720) | (960, 720) |
| (695, 460) | (960, 0) |

<br>Following is an example of a warped image:
![alt text][warp]

### Lane Detection
Fisrt, I defined left and right lane x-positions as the two most prominent peaks of a histogram. The histogram was calculated by adding up all the pixel values in columns of a image.

![alt text][histogram]

Then, I did window searching from the bottom of the image with the number of windows 9, the height of each window 1/9 of the image and the width 2 margin. The margin is 100. For each window, I picked up the points within the window. I fit a second order polynomial function to these points and defined the lines drawn by the polynomial function as the lanes detected.
The codes for this step are in `pipeline.py`.
<br>Following is an example of a image of the lane detection:
![alt text][lane]

### Measuring Radius of Curvature
According to [this page](http://www.intmath.com/applications-differentiation/8-radius-curvature.php), I calculated the radius of curvature converting the coordinates in pixel space to the coordinated in real world space.
<br>This step was done in the `# Curvature` section in `scripts/pipeline.py`.

### Output
I combined all the steps above and got images like below.
![alt text][output]

### Pipeline (video)
Here's a [link to my video result](./project_video_output.mp4)
