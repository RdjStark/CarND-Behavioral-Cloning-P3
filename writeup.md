# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[center1]: ./writeup_images/center1.jpg "Drive in the center 1"
[center2]: ./writeup_images/center2.jpg "Drive in the center 2"
[recovery1]: ./writeup_images/recovery1.jpg "Recovery starts - dirt 1"
[recovery2]: ./writeup_images/recovery2.jpg "Recovery starts - dirt 2"
[recovery3]: ./writeup_images/recovery3.jpg "Recovery starts - line 1"
[recovery4]: ./writeup_images/recovery4.jpg "Recovery starts - line 2"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md summarizing the results
* writeup_images a folder containing writeup images
* lapfinal.mp4 Video showing autonomous driving one and half lap on track 1

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5 lapfolder
```
#### 3. Submission code is usable and readable
model.py is the submission code. It contains codes for reading and preprocessing data, defining the deep learning architecture and  training it. The code is well commented and works. 

---
### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model was defined in model.py between code lines #93 and #149 based on LeNet-5 architecture. It contains two 5x5 convolution layers, three fully connected layers and other building blocks: max pooling layers, dropout layers, activation functions. The input is preprocessed with a normalizing lambda function and a cropping function.

#### 2. Attempts to reduce overfitting in the model

To help the model against overfitting, I placed a dropout layer with dropout rate of 20% in code line #140. It is injected between two fully connected layers. 

The driving data used for training is splitted into two parts in code line #22. Training data has 80%, while validation data has 20% of the elements.

#### 3. Model parameter tuning

I was using Adam Optimizer, so model parameters (e.g. learning rate) were not set explicity.

#### 4. Appropriate training data

Training data were created mixing my driving data and the preinstalled data on the workspace. My data are focusing mainly on recovering and the tricky parts.

---
### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My strategy was not building a model from scratch but to find a model that is appropriate for this purpose. I have chosen LeNet-5 because CNN's are good for image recognition.

First my model was overfitting. To reduce overfitting, I placed a dropout layer between the dense layers. Also collecting more data, augmenting it by flipping and splitting the data 20% - 80% between training and validating data helped a lot.

A problem raised when I started to use left and right camera images. Because of the badly aligned left, right measurement multipliers, even with triple data, my model's training and validation losses were three times bigger than using data just from center camera and the car was waddling slowly and finally left the road. 

There were no activation functions in the dense layers of the base architecture, so it put them there to improvde nonlinearity.  

First using my model in simulator caused the car wander off the road in these scenarios:
  * Car slowly moved to the side of the road. Data were appended showing my model how to recover from it
  * After the bridge, there is a brown dirt to the right. Model had to train not to leave the road here with numerous recovery records.
  
At the end of the process, the vehicle was able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

A table shown below describes my final architecture 

|Layer|Description|
|:---:|:---------:|
|Input|320x65x3 RGB image|
|Convolution|1x1 stride, valid padding, outputs 5x5x8|
|Activation function|RELU|
|Max pooling|2x2 stride, valid padding|
|Convolution|1x1 stride, valid padding, outputs 5x5x6|
|Activation function|RELU|
|Max pooling|2x2 stride, valid padding|
|Fully connected|outputs 120|
|Activation function|RELU|
|Dropout|20%|
|Fully connected|outputs 84|
|Activation function|RELU|
|Fully connected|outputs 1|

#### 3. Creation of the Training Set & Training Process

I used a mix of previous recorded images and my own driving behaviour, I recorder two laps by myself paying attention to drive in the center. <br/> 
Here are some examples:<br/>
![alt text][center1]
![alt text][center2]

Then I recorder recovery maneuvers. 

For example, turning back before going off-road...

![alt text][recovery1]
![alt text][recovery2]

...or pulling back from the side.

![alt text][recovery3]
![alt text][recovery4]


After the collection process, I had 20245 center images: 8036 from the preinstalled set and 12209 from my drivings. 
I then preprocessed this data by flipping it on the y axis, so the final number is 
40490. It was shuffled on the begining of each process, and data was split 20/80 between validation and training set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 8, because validation loss started to flat out at this point, as seen below:

<pre>
Epoch 1 254/254 55s 216ms/step - loss: 0.0205 - val_loss: 0.0140
Epoch 2 254/254 52s 205ms/step - loss: 0.0123 - val_loss: 0.0127
Epoch 3 254/254 51s 202ms/step - loss: 0.0105 - val_loss: 0.0109
Epoch 4 254/254 52s 206ms/step - loss: 0.0095 - val_loss: 0.0111
Epoch 5 254/254 51s 201ms/step - loss: 0.0087 - val_loss: 0.0098
Epoch 6 254/254 52s 206ms/step - loss: 0.0081 - val_loss: 0.0096
Epoch 7 254/254 52s 205ms/step - loss: 0.0075 - val_loss: 0.0092
Epoch 8 254/254 52s 204ms/step - loss: 0.0068 - val_loss: 0.0092
</pre>

I set batch size to 64, set epoch number to 8 and used an Adam Optimizer, so that manually training the learning rate wasn't necessary.
