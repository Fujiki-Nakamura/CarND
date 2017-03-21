## Vehicle Detection Project Report
[//]: # (Image References)
[image1]: ./output_images/vehicle_vs_nonvehicle.jpg
[image2_1]: ./output_images/HOG_vehicle.jpg
[image2_2]: ./output_images/HOG_nonvehicle.jpg
[image3]: ./output_images/visualize_sliding_window.jpg
[image4]: ./output_images/vehicle_detection_sliding_window.jpg
[image5]: ./output_images/pipeline.jpg

### Goals and steps
The goals / steps of this project are the following:
* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.


### Histogram of Oriented Gradients (HOG)
The code for this step is `get_hog_features()` function in `utils/feature_extraction.py`.

#### HOG
I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`). I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

HOG image xamples using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)` are following:

Vehcile
![alt text][image2_1]
Non Vehicle
![alt text][image2_2]

Finally, I settled on HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`. With other 2 parameters fixed (i.e. `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`), I tried several `orientations`. Increasing `orientations` increased the time of training the model while not improving the accuracy of the model so much. So, I finally chose `orientaions=9`.

#### Model training
The code for this step is [`model_training.ipynb`]('model_training.ipynb').

I trained a linear SVM with HOG features, color histogram features and bin spatial features. The color space of images fed to the model was `YCrCb`. With `YCrCb` color space, the model best performed.

Below are the parameters for getting each features.

| Feature type | Parameter | Value |
| --- | --- | --- |
| HOG | orientations | 9 |
|  | pixels_per_cell | (8, 8) |
|  | cells_per_block | (2, 2) |
|  | channel | ALL |
| Color histogram | hist bins | 32 |
| Bin spatial | spatial size | (32, 32) |

After splitting the images into train images and test images in a stratified way, I did 5-Fold cross validation with the train images. The mean validation accuracy was `99.20%`, which was a pretty good result. Then I tested the model trained with whole train images against the test images. The test accuracy was `99.03%`, and again I got a pretty good result. So, I save the model to use it for the sliding window search step.


### Sliding Window Search
The code for this step is  [`vehicle_detection.ipynb`]('vehicle_detection.ipynb').

#### Visualixe Sliding Windows
This step was done in the section [`Draw Sliding Window`]('vehicle_detection.ipynb#Draw-Sliding-Windows') in  [`vehicle_detection.ipynb`]('vehicle_detection.ipynb') using the function `draw_sliding_windows()` in `draw_functions.py`.

The windows have `75%` overlap by setting `cells_per_step=2`. I tried several combinations of `scale`, `ystart` and `ystop`, and empirically decided 3 different combinations. Below are the table of the combinations and the image to visualize the sliding windows of each combination.

| scale | ystart | ystop |
| --- | --- | --- |
| 375 | 475 | 1 |
| 400 | 600 | 1.5 |
| 500 | 700 | 2.5 |

![alt text][image3]

#### Vehicle Detection by Sliding Window Search
This step was done in the section [Vehicle Detection]('vehicle_detection.ipynb#Vehicle-Detection') in [`vehicle_detection.ipynb`]('vehicle_detection.ipynb') using the function `find_cars()` in `feature_extraction.py`.

I seached on the 3 scales described above using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector. The example result is like below:

![alt text][image4]


### Video Implementation
Here's a [link to my video result](./project_video_output.mp4)

#### Heatmap and Label Detection
This step was done in the section [`Heatmap`]('vehicle_detection.ipynb#Heatmap') and [`Label Detection`]('vehicle_detection.ipynb#Label-Detection') in `vehicle_detection.ipynb`.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap. I then assumed each blob corresponded to a vehicle. I constructed bounding boxes to cover the area of each blob detected.  

An example image of this step is below:

![alt text][image5]


### Discussion
- To make the search more robust, I could try more combinations of `scale`, `ystart`, and `ystop`. Also, higher thresholds might be useful to avoid false positives.
- Using information from several previous frames could make the detection more robust.
- A Neural Network classifier might be more powerful to detect cars in sliding window search.
