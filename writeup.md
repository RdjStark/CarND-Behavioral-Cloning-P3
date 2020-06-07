# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

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

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5 lapfolder
```
#### 3. Submission code is usable and readable
model.py is the submission code. It contains codes for reading and preprocessing data, defining the deep learning architecture and  training it. The code is well commented and works. 

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model was defined in model.py between code lines #93 and #149 based on leNet-5 architecture. It contains two 5x5 convolution layers, three fully connected layers and other building blocks: max pooling layers, dropout layers, activation functions. The input is preprocessed with a normalizing lambda function and a cropping function.

#### 2. Attempts to reduce overfitting in the model

To help the model against overfitting, I placed a dropout layer with dropout rate of 20% in code line #140. It is injected between two fully connected layers. 

The driving data used for training is splitted into two parts in code line #22. Training data has 80%, while validation data has 20% of the elements.

#### 3. Model parameter tuning

I was using Adam Optimizer, so model parameters were not set explicity.

#### 4. Appropriate training data

Training data were created mixing my driving data and the preinstalled data on the workspace. My data are focusing mainly on recovering and the tricky parts.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My strategy was not building a model from stratch but to find a model that appropriate for this purpose. I chose LeNet-5 because CNNs are good for image recognition.

First my model was overfitting. To reduce overfitting is placed a dropout layer between the dense layers. Also collecting more data, augmenting it by flipping and splitting the data 20% - 80% between training and validating data helped a lot.

A problem raised when i started to use left and right camera images. Because of the badly aligned left, right measurement multipliers, even with triple data, my model training loss and validations loss were three times bigger than using data just from center camera and the car was waddling slowly and left the road at the begining. 

There were no activation functions in the dense layers, so it put them there to improvde nonlinearity.  

First using my model in simulator caused the car wander off the road in these scenarios:
  * Car slowly moved left to right to the side on straigth roads. Data were appended showing my model how to recover from it
  * After the bridge, there is a brown dirt to the right. Model had to train not tp leave the road here with numerous recovery records.
  
At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture 

|Layer|Description|
|:---:|:---------:|
|Input|320x65x3 RGB image|
|Convolution|1x1 stride, valid padding, Outputs 5x5x8|
|Activation function|RELU|
|Max pooling|2x2stride valid padding|
|Convolution|1x1 stride, valid padding, Outputs 5x5x6|
|Activation function|RELU|
|Max pooling|2x2stride valid padding|
|Fully connected|Outputs 120|
|Activation function|RELU|
|Drop out|20%|
|Fully connected|Outputs 84|
|Activation function|RELU|
|Fully connected|Outputs 1|



#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
