## Behavioral Cloning Project Report
[image1]: ./images/nvidia_network.png "Model Visualization"
[image2]: ./images/center_images.jpg
[image3]: ./images/steering_histogram.jpg
[image4]: ./images/left_center_right.jpg

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

### Model Architecture and Training Strategy
#### An appropriate model arcthiecture
The model I used is coded in `model_nvidia_like.py`. It has 5 convolutional layers followed by 4 fully connected layers. The activation function for introducing nonlinearity is ReLU. The model preprocess the images in its first few layers. It crops the images and obtains the region of interest of the each image. Then it normalizes the cropped images for the future layers.

#### Attempts to reduce overfitting in the model
The model contains dropout layers in its first three fully connected layers to avoid overfitting. To see if the model is overfitting, the data was splitted into the training data and the validation data. The model was trained on the training data and then validated on the validation data.

#### Model parameter tuning
I used an adam optimizer for the model, so I didn't need to tune the learning rate manually. This is coded in `model_nvidia_like.py`.

#### Training data
I used the data provided by Udacity. In addition to the center images, I used the left and right images with some corrections for steering angles so that the model learns recovering from the left and right sides of the road.

The detail is in the next section.


### Architecture and Training Documentation
#### Solution Design
First, I implemented a model which looks like the NVIDIA model. This model seemed appropriate because the model was designed for solving a problem similar to this project and performed successfully. The model is described in [the paper](https://arxiv.org/pdf/1604.07316v1.pdf). I modified this model a little and I added one more fully connected layer.

In order to see how well the model predicts the steering angles, I trained it with 75% of the training data and validated it with 25% of the training data. The training data consisted only of the center images. The training loss and the validation loss are both between 0.010 and 0.015. It seemed that the model predicted well, so I checked the performance of the model on the simulator.

In the simulator, the car with the model drived well on almost all the parts of the cource. However, it fell off the track sometimes , especially at some corners.

In order to keep the car on the track throughout the cource, I used the left and right images in the training process with some corrections for the center steering angles. This step is described in the section later.

After learning how to recover from the left or right sides of the road, the autonomous car with the model was able to drive by itself around the track 1.

#### Final Model Architecture
The final model architecture is similar to the NVIDIA model. The NVIDIA model looks like below. It has 5 convolutional layers and 4 fully connected layers.

![alt text][image1]

Based on this model, the final model was implemented like below:
- Layer 1: Convolutional feature map 24@5×5
- Layer 2: Convolutional feature map 36@5×5
- Layer 3: Convolutional feature map 48@5×5
- Layer 4: Convolutional feature map 64@3×3
- Layer 5: Convolutional feature map 64@3×3
- Layer 6: Fully-connected layer, 64 units
- Layer 7: Fully-connected layer, 32 units
- Layer 8: Fully-connected layer, 16 units
- Layer 9: Fully-connected layer, 8 units
- Layer 10: Output layer

#### Creation of the Training Set & Training Process
I finally found that the data provided by Udacity are enough for the model building. So, I decided to use the data only and not to use the data that I collected. For I thought I was not good at driving in the simulator and the collected data might have some inappropriate samples.

Several examples of the images recorded by the center camera are below. Each image is show with its coressponding steering angle.

![alt text][image2]

As shown in the examples above, the data seemed to have a lot of `steering angle = 0.0` and few other steering angles. I confirmed that by drawing the histogram of the steering angles. As shown in the histogram below, the data has a lot of steering angle around 0, especially slightly less than 0. On the other hand, the data has few steering angles other than that, especially less than -0.25 or more than 0.25. Because of this, the model trained only with the center images and the corresponding original steering angle is likely to go straight and not likely to learn how to recover when it is on the left or right sides of the road. In fact, in my experiment where I only used the center images, the vehicle was not able to steer sharply and fell off the track.

![alt text][image3]

In order to teach the car how to keep itself on the track throughout the cource and how to avoid falling of the track, I augmented the traing data by using the left and right images. For the left images, I defined their steering values as `the steering value of center image + 0.2`. In the same way, I defined the steering values of the right images as `the steering value of center image - 0.2`. The parameter `0.2` was chosen arbitrarily but I found that the value was enough for the car to drive throuthout the cource without leaving off the road.

Below are some examples of the augmentation.

![alt text][image4]
