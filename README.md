
---

**Behavioral Cloning Project**

Overview
---
This repository contains starting files for the Behavioral Cloning Project.

In this project, a deep neural networks and convolutional neural networks was deployed to clone driving behavior, this will 
be used to train, validate and test a model using Keras. The model will output a steering angle to an autonomous vehicle.

Udacity has provided a simulator where a car can be steered around a track for data collection. Image data and steering angles are used to train a neural network and then use this model to drive the car autonomously around the track.


The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/original_image.png "Normal Image"
[image2]: ./examples/flip_image.png "Flipped Image"
[image3]: ./examples/original_image_geo_shifted.png "Geo Shifted Image"

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode (this is modified to adjust the throttle)
* model.h5 containing a trained convolution neural network for cloning the driving behaviour on TRACK 1 
* model_track_2.h5 containing a trained convolution neural network for cloning the driving behaviour on TRACK 2
* run5.mp4 video to drive the car on Track 1
* run4.mp4 video to drive the car on Track 2
* README.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around both track by executing 
```sh
python3 drive.py model.h5
```
and for track 2, see the comment on drive.py and section 'Conclusion' at the end of this document and run the command bellow.

```sh
python3 drive.py model_track_2.h5   
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

First I applied a very basic network with just a flatten layer and an output unit of 1 to make sure everything is working in a sense of reading the images and the corresponding steering angles and use keras to fit the model; as this is a regression network instead of a classification network, mean squar error was used as the loss function instead of cross entropy. Idea is to minimize the difference between the steering measurement this network predicts and the ground truth steering measurement, I shuffle the data and use 20% of the data for validation set.

Then I tried with LeNet CNN network which takes 32x32x1 image only but as the Convolutional layer from keras works with wide ranges of images, I able to use the images
here which are 160x320x3 without resizing them.

Finally I moved to PilotNet CNN published by NVIDIA. 


#### 2. Attempts to reduce overfitting in the model

The model was modified with a dropout layers in order to reduce overfitting by experiemnt on both track. Also it was trained and validated on different data sets to ensure that the model was not overfitting (see Data augmentation i.e. flip the image during cornering, steering angle adjustment, applied Gausin Blur/DownSample the image etc.). The model was tested by running it through the simulator and ensuring that the vehicle could stay and run smoothly on the track 1.
 
For track 2 I just make sure the car was on the road (either on left, middle and right lane) not falling off the track, this is due to less available data on this track (I trained it just enough to be on the road for the full track due to time constrain of this project submission)

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination driving of clockwise and counter-clockwise; use side (left and right) cameras to train the car how to recover the bias from left or right during cornering.For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a convolution neural network model similar to leNet, I thought this model might be appropriate and it is simple but then I learned as part of this self driving course that leNet was developed within the mind set of classifying digits and can take only 32x32 pixel images, the ability to process higher resolution images require larger and more convolutional layers and I found this model was overfitting during the validation.

In contract, PilotNet from Nvidia was developed on the base to find the region in input images which makes the steering decisions, they call it the sailent objects, structure in camera images those are not relevant to driving, as they described in the paper here, https://arxiv.org/pdf/1704.07911.pdf, also this capability is derived from data without the need of hand-crafted rules. 

In this self driving regression network we wanted to minimize the mean square error instead of reducing the cross-entropy which is for the classifying network and I decided to use PilotNet which gave better result in the given project.


In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set(20%). The network must also learn how to recover from any deviation or the car will drift off the road slowly. The training data is therefore augmented with additional images from left and right camera which shows the car in different shifts from the center of the lane and rotations from the direction of the road.

At the end of the process, the vehicle is able to drive autonomously around both track without leaving the road.

####2. Final Model Architecture

I used the  NVIDIA'S PilotNet cnn network with an extra dropout layer added after the convolutional layers.
According to NVIDIA, the convolutional layers are designed to perform feature extraction, and are chosen empirically  through a series of experiments that vary layer configurations. They use strided convolutions in the first three convolutional layers with a 2×2 stride and a 5×5 kernel, and a non-strided convolution with a 3×3 kernel size in the final two convolutional layers.

I found it helped reducing the overfitting of these model in the given track by adding a dropout layer after the flatten
layer. Filters depth in this network is between 24 to 64.

Here is a visualization of the Architecture

Layer(type)                 Output Shape              Param#     

lambda-1 (Lambda)            (None, 160, 320, 3)       0         
cropping2d-1 (Cropping2D)    (None, 65, 320, 3)        0         
conv2d-1 (Conv2D)            (None, 31, 158, 24)       1824      
conv2d-2 (Conv2D)            (None, 14, 77, 36)        21636     
conv2d-3 (Conv2D)            (None, 5, 37, 48)         43248     
conv2d-4 (Conv2D)            (None, 3, 35, 64)         27712     
conv2d-5 (Conv2D)            (None, 1, 33, 64)         36928     
flatten-1 (Flatten)          (None, 2112)              0         
dropout-1 (Dropout)          (None, 2112)              0         
dense-1 (Dense)              (None, 100)               211300    
dense-2 (Dense)              (None, 50)                5050      
dense-3 (Dense)              (None, 10)                510       
dense-4 (Dense)              (None, 1)                 11        
Total params: 348,219
Trainable params: 348,219
Non-trainable params: 0

 The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a
   Keras lambda layer (code line 188). 

I prefer to use RELU instead of ELU based on the recent experiment bellow, https://ctjohnson.github.io/Capstone-Activation-Layers/
which summarized the comparison of different activation layers
as bellow,

   Activation   Test Accuracy   Training Time (per epoch)
     Relu          90.25%           76s
     Elu           84.36%           76s
     Soft Plus     2.85%            78s
     TanH          83.41%           78s
     Sigmoid       5.7%             90s
     Linear        83.6%            131s
     Soft Sign     85.95%           82s
     Hard Sigmoid  5.7%             85s

#### 3. Creation of the Training Set & Training Process

On Track 1,

To capture good driving behavior, I first recorded two laps  using clockwise and counter-clockwise. To reduce the bias towards cornering I added a small offset to the left side camera images and deducted from the right camera images and later on if the steering angle is more than 0.43, a flipped image was feed to the network to recover from over drifting.
See the section 'Data Augmentation' bellow for details.

As explaind in Udacity course material i.e. "Explanation of How Multiple Cameras Work", It is not necessary to use the left and right images to derive a successful model, Recording recovery driving from the side of the road is also effective. I took the first approach and used the left and right camera images to record the recovery.


Epoch 1/7
13343/13387 [============================>.] - ETA: 0s - loss: 0.0333
13408/13387 [==============================] - 34s - loss: 0.0332 - val_loss: 0.0253
Epoch 2/7
13429/13387 [==============================] - 27s - loss: 0.0261 - val_loss: 0.0239
Epoch 3/7
13427/13387 [==============================] - 27s - loss: 0.0241 - val_loss: 0.0224
Epoch 4/7
13418/13387 [==============================] - 27s - loss: 0.0226 - val_loss: 0.0235
Epoch 5/7
13426/13387 [==============================] - 27s - loss: 0.0218 - val_loss: 0.0223
Epoch 6/7
13424/13387 [==============================] - 27s - loss: 0.0209 - val_loss: 0.0209
Epoch 7/7
13429/13387 [==============================] - 27s - loss: 0.0203 - val_loss: 0.0213

I can see the training and validation loss was reduced over 6 epochs until the last one where the validation loss was incresed a bit to indicate any more iteration will overfit the model.

On Track 2,

I just train the car once for the full track, counter clockwise and added 2 or 3 extra images for the cornering where the car was failing of the road, increase the number of epoch untill the loss was decreased and seems like PilotNet was able to run the car on track even without doing any breaking or further data.

16538/16593 [============================>.] - ETA: 0s - loss: 0.1628 
16624/16593 [==============================] - 46s - loss: 0.1627 - val_loss: 0.1537
Epoch 2/9
16655/16593 [==============================] - 33s - loss: 0.1272 - val_loss: 0.1168
Epoch 3/9
16618/16593 [==============================] - 32s - loss: 0.1155 - val_loss: 0.1120
Epoch 4/9
16617/16593 [==============================] - 32s - loss: 0.1056 - val_loss: 0.1016
Epoch 5/9
16610/16593 [==============================] - 32s - loss: 0.0978 - val_loss: 0.1004
Epoch 6/9
16606/16593 [==============================] - 32s - loss: 0.0952 - val_loss: 0.0950
Epoch 7/9
16616/16593 [==============================] - 32s - loss: 0.0891 - val_loss: 0.0951
Epoch 8/9
16620/16593 [==============================] - 32s - loss: 0.0843 - val_loss: 0.0903
Epoch 9/9
16603/16593 [==============================] - 32s - loss: 0.0807 - val_loss: 0.0918

Specially for track 2, converting the image from RGB to YUV (as suggested by Nvidia) and downsample it (by applying 
a Gaussian Blur) makes the car drive much smoother.

It is also imparitive to do the same conversion on drive.py where the image was received from the simulator in RBG
format.

1. Data Augmentation:
 
Cropping the Image
 I use keras Cropping2D function to Crop the tree, hills, Sky from top 70 and the hood of the car from bottom 25
to avoid any extra noise on the fly.

Multiple Camera Images
 There are multiple cameras on the vehicle, and I used map recovery paths from each camera. For example, while the model was  trained to associate a given image from the center camera with a left turn, it was also trained to associate the corresponding image from the left camera with a somewhat softer left turn and to associate the corresponding image from the right camera with an even harder left turn. This is applied on line 63 to 73 of model.py. This way, the car has been taught how to steer if the car drifts off to the left or the right

Flipping the Images
 I added more data by flipping the images for those where the absolute value of steering is > 0.43 (by experiment this value was selected) to prevent overfitting . An effective technique for helping with the left or right turn bias involves flipping images and taking the opposite sign of the steering measurement. For example, here is an image that has then been flipped:

![alt text][image1]
![alt text][image2]


Geometric Translation of an Image
 I shifted the original images on right, left , up and down but after experiment I did not find this helpfull with 
  the training data I collected, one of the reason could be when I shift the image, it replaced the shifted location with
  a black backdround which I should avoid by resizing the image. 	

![alt text][image1]
![alt text][image3]


After the collection process, I had for Track 1, 
13387 images for training and 3347 images for validation

and for Track 2,
16593 images for training and 4149 for validation.

I randomly shuffled the data and used python generator to feed the data into the model in a memory optimized
batched fashion

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 6 for track 1 and 8 for track 2 within my training set as evidenced in section
 3.'Creation of the Training Set & Training Process' above.  Adam optimizer was used to avoid manual training of the learning rate.

#### 4. Conclusion
I tried to keep the network simple like the pilotNet while cover both track. I would say Udacity 'Self Driving' course material for deep neural network and convolutional neural networks was quite good enough to clone the driving behaviour on both track successfully. It was also hinted where I could have make the final layer to output 3 things; the Steering Angle, speed and break measurement which then I could use to drive the car more dynamically with different speeds; faster in the case of straight road or slower by applying the break on a downhill situation. I also found it by adding a dropout layer allow me to train the network further and produce a stable driving. For track 2 I could have train the model to not only stay on road but also on track by adding more datapoints but due to time limitation I just trained it enough to be on road for track 2 for the fulllaps.

I also like to add, AWS gpu support is essential specially for track 2 to tune the model optimally within the confined 
time limit. I observed  dramatic increase in computing performace while using CUDA library invented by NVIDIA on AWS's gpu
compare to tensor flow 2.0 or above running on my Intel cpu. 

