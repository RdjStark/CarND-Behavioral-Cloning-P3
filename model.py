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
epoch_number = 6

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
# Convolution:      5x5x24 stride 2,2                  #
# RELU                                                 #
# Convolution:      5x5x36 stride 2,2                  #
# RELU                                                 #
# Convolution:      5x5x48 stride 2,2                  #
# RELU                                                 #
# Convolution:      3x3x64 stride 1,1                  #
# RELU                                                 #
# Convolution:      3x3x64 stride 1,1                  #
# RELU                                                 #
# Dropout           20%                                #
# Fully connected   outputs 100                        #
# Fully connected   outputs 64                         #
# Fully connected   outputs 16                         #
# Fully connected   outputs 1                          #
########################################################

# Started from an empty sequential model
model = Sequential()

# A lambda fuction was used to preprocess the data. It was normalized to -0.5 .. 0.5
model.add(Lambda(lambda x: (x/255.0) -0.5, input_shape=(160,320,3)))
# The upper and lower parts of each image are rather confusing than useful. 
# These parts were cropped
model.add(Cropping2D(cropping=((70,25),(0,0))))
# A convolution layer with 5*5 filter, stride is 2,2 Output depth is 24
model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
# A convolution layer with 5*5 filter, stride is 2,2 Output depth is 36
model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
# A convolution layer with 5*5 filter, stride is 2,2 Output depth is 48
model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
# A convolution layer with 3*3 filter, stride is 1,1 Output depth is 64
model.add(Convolution2D(64,3,3, activation='relu'))
# A convolution layer with 3*3 filter, stride is 1,1 Output depth is 64
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Flatten())
model.add(Dropout(0.2))    
# A fully connected layer followed by a relu activation function
model.add(Dense(100))
# A fully connected layer followed by a relu activation function
model.add(Dense(64))
# A fully connected layer followed by a relu activation function
model.add(Dense(16))
# A fully connected layer followed by a relu activation function
model.add(Dense(1))
    
# An Adam Optimizer closes the pipeline using mean square error
model.compile(loss='mse', optimizer='adam')

# Train the model. 
model.fit_generator(train_generator, 
                    steps_per_epoch=math.ceil(6 * len(train_lines)/batch_size) - 1,
                    validation_data=valid_generator,
                    validation_steps=math.ceil(6 * len(valid_lines)/batch_size) - 1,
                    epochs=epoch_number,
                    verbose=1)
                   
model.save('model.h5')
