import os
import csv
import cv2
import sklearn
import numpy as np
from random import shuffle
import math
import cv2
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Convolution2D, MaxPooling2D, Dropout
from sklearn.model_selection import train_test_split        

# Read the file containing links to camera photos and measured data
lines = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
# Split the data into two sets, into a training set and a valiadion set.
# Split rate is 20%         
train_lines, valid_lines = train_test_split(lines, test_size = 0.2)

# Set the batch size of processing of training and validation data 
batch_size = 64

# Set number of the consecutive and related training runs
epoch_number = 8

# Define a generator that process images in pieces instead of 
# reading the whole data It uses much less memory then the bulk process. 
# 
def sample_generator(lines, batch_size=64, side_camera_correction=0.15):    
    num_lines = len(lines)
    while 1:       
        
        # Shuffle the data before every epoch
        shuffle(lines)
        batch_count = num_lines // batch_size
        
        # process one batch
        for offset in range(0, batch_count, 1):
            images = []
            measurements = []
            
            # Sample is the list of data (image links and measurements) in one batch 
            sample = lines[offset * batch_size:(offset+1) * batch_size]
            for item in sample:   
                try:
                    # Read a center camera image
                    center_image_name = 'data/IMG/' + item[0].split('/')[-1]
                    center_image = cv2.imread(center_image_name)   
                    
                    # Save the X (center camera image) and y (measurement to be compared to) being processed
                    images.append(center_image)
                    measurements.append(np.float(item[3]))                   
                    
                    # Augment (double) the data with flipping on the y axis. 
                    images.append(cv2.flip(center_image,1))
                    measurements.append(-1.0 * np.float(item[3]))                                         
                    
                    left_image_name = 'data/IMG/' + item[1].split('/')[-1]
                    left_image = cv2.imread(left_image_name)   
                    images.append(left_image)                    
                    measurements.append(np.float(item[3]) + side_camera_correction)  
                    images.append(cv2.flip(left_image,1))
                    measurements.append(-1.0 * (np.float(item[3]) + side_camera_correction)) 

                    right_image_name = 'data/IMG/' + item[2].split('/')[-1]
                    right_image = cv2.imread(right_image_name)   
                    images.append(right_image)
                    measurements.append(np.float(item[3]) - side_camera_correction)                
                    images.append(cv2.flip(right_image,1))
                    measurements.append(-1.0 * (np.float(item[3]) - side_camera_correction)) 
                    

                except:
                    print("Warning: item=(" + item +")")
            
            yield(np.array(images), np.array(measurements))

# Setup the generators for training data and validation data            
train_generator = sample_generator(train_lines, batch_size = batch_size)
valid_generator = sample_generator(valid_lines, batch_size = batch_size)

#(X_train, y_train) = next(train_generator) # testing purpose

########################################################
# A LeNet-5 architecture was built:                    #
#                                                      #
# Input:            320x65x3                           #
# Convolution:      5x5x8                              #
# RELU                                                 #
# Maximum Pooling   2x2 valid padding                  #
# Convolution:      5x5x6                              #
# RELU                                                 #
# Maximum Pooling   2x2 valid padding                  #
# Fully connected   outputs 120                        #
# RELU                                                 #
# Dropout           20%                                #
# Fully connected   outputs 84                         #
# RELU                                                 #
# Fully connected   outputs 1                          #
########################################################

# Started from an empty sequential model
model = Sequential()

# A lambda fuction was used to preprocess the data. It was normalized to -0.5 .. 0.5
model.add(Lambda(lambda x: (x/255.0) -0.5, input_shape=(160,320,3)))
# The upper and lower parts of each image are rather confusing than useful. 
# These parts were cropped
model.add(Cropping2D(cropping=((70,25),(0,0))))

# A convolution layer with 5*5 filter. Output depth is 8. Activation function is a rectifier
# linear unit
model.add(Convolution2D(8,5,5,activation="relu"))

# A maximum pooling layer
model.add(MaxPooling2D())

# A convolution layer with 5*5 filter. Output depth is 6
model.add(Convolution2D(6,5,5,activation="relu"))

# A maximum pooling layer used against overfitting
model.add(MaxPooling2D())

# Reshape to 1 dimension
model.add(Flatten())

# A fully connected layer followed by a relu activation function
model.add(Dense(120, activation="relu"))

# A dropout layer prevents overfitting
model.add(Dropout(0.2))

# A fully connected layer followed by a relu activation function
model.add(Dense(84, activation="relu"))

# A fully connected layer without activation function
model.add(Dense(1))

# An Adam Optimizer closes the pipeline using mean square error
model.compile(loss='mse', optimizer='adam')

# Train the model. 
model.fit_generator(train_generator, 
                    steps_per_epoch=math.ceil(len(train_lines)/batch_size),
                    validation_data=valid_generator,
                    validation_steps=math.ceil(len(valid_lines)/batch_size),
                    epochs=epoch_number,
                    verbose=1)
                   
model.save('model.h5')
