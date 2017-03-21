## Behavioral Cloning project

### Model Architecture and Training Strategy
#### An appropriate model arcthiecture
The model I used is coded in `model_nvidia_like.py`. It has 5 convolutional layers followed by 4 fully connected layers. The activation function for introducing nonlinearity is ReLU. The model preprocess the images in its first few layers. It crops the images and obtains the region of interest of the each image. Then it normalizes the cropped images for the future layers.

#### Attempts to reduce overfitting in the model
The model contains dropout layers in its first three fully connected layers to avoid overfitting. To see if the model is overfitting, the data was splitted into the training data and the validation data. The model was trained on the training data and then validated on the validation data.

#### Model parameter tuning
I used an adam optimizer for the model, so I didn't need to tune the learning rate manually. This is coded in `model_nvidia_like.py`.

#### Training data
In addition to the data which I collected with the simulator, I used the data provided by Udacity. In order to train the model, I used all the kind of data, center, left and right.


### Architecture and Training Documentation
#### Solution Design
First, I implemented a model which looks like the NVIDIA model. This model seemed appropriate because the model was designed for solving a problem similar to this project and performed successfully. The model is described in [the paper](https://arxiv.org/pdf/1604.07316v1.pdf). I modified this model a little and I added one more fully connected layer.

In order to see how well the model predicts the steering angles, I trained it with 75% of the training data and validated it with 25% of the training data. The training data consisted only of the center images. The training loss and the validation loss are both between 0.010 and 0.015. It seemed that the model predicted well, so I checked the performance of the model on the simulator.

In the simulator, the car with the model drived well on almost all the parts of the cource. However, it fell off the track at some corners.

In order to keep the car on the track throughout the cource, I used the left and right images in the training process. This step is described in the section later.

At the end of the process, the autonomous car with the model was able to drive by itself around the track 1.

#### Final Model Architecture
The final model architecture is similar to the NVIDIA model. The NVIDIA model looks like below. It has 5 convolutional layers and 4 fully connected layers.
[image1]: ./images/nvidia_network.png "Model Visualization"
![alt text][image1]

Based on this model, the final model is implemented like below:
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

In order to teach the car how to keep itself on the track throughout the cource and how to avoid falling of the track, I augmented the traing data by using the left and right images. For the left images, I defined their steering values as `the steering value of center image + 0.2`. In the same way, I defined the steering values of the right images as `the steering value of center image - 0.2`. The parameter `0.2` was chosen arbitrarily but I found that the value was enough for the car to drive throuthout the cource without leaving the road.
