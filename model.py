
import os
import csv
import cv2
import sklearn
import numpy as np
from random import shuffle
import math
import cv2

lines = []
with open('/home/workspace/driving_data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
from sklearn.model_selection import train_test_split        
train_lines, valid_lines = train_test_split(lines, test_size = 0.2)

batch_size = 64

def sample_generator(lines, batch_size=64, side_camera_correction=0.3):    
    num_lines = len(lines)
    while 1:        
        shuffle(lines)
        batch_count = num_lines // batch_size
        for offset in range(0, batch_count, 1):
            images = []
            measurements = []
            sample = lines[offset * batch_size:(offset+1) * batch_size]
            for item in sample:   
                try:
                    center_image_name = '/home/workspace/driving_data/IMG/' + item[0].split('/')[-1]
                    center_image = cv2.imread(center_image_name)   
                    images.append(center_image)
                    measurements.append(np.float(item[3])) 
                    images.append(cv2.flip(center_image,1))
                    measurements.append(-1.0 * np.float(item[3])) 
                    """
                    left_image_name = '/home/workspace/driving_data/IMG/' + item[1].split('/')[-1]
                    left_image = cv2.imread(left_image_name)   
                    images.append(left_image)                    
                    measurements.append(np.float(item[3]) + side_camera_correction)  
                    images.append(cv2.flip(left_image,1))
                    measurements.append(-1.0 * (np.float(item[3]) + side_camera_correction)) 

                    right_image_name = '/home/workspace/driving_data/IMG/' + item[2].split('/')[-1]
                    right_image = cv2.imread(right_image_name)   
                    images.append(right_image)
                    measurements.append(np.float(item[3]) - side_camera_correction)                
                    images.append(cv2.flip(right_image,1))
                    measurements.append(-1.0 * (np.float(item[3]) - side_camera_correction)) 
                    """

                except:
                    print("Warning: item=(" + item +")")

                
            yield(np.array(images), np.array(measurements))
                         
train_generator = sample_generator(train_lines, batch_size = batch_size)
valid_generator = sample_generator(valid_lines, batch_size = batch_size)

(X_train, y_train) = next(train_generator)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Convolution2D, MaxPooling2D, Dropout

model = Sequential()
model.add(Lambda(lambda x: (x/255.0) -0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))

model.add(Convolution2D(8,5,5,activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(6,5,5,activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(84, activation="relu"))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, 
                    steps_per_epoch=math.ceil(len(train_lines)/batch_size),
                    validation_data=valid_generator,
                    validation_steps=math.ceil(len(valid_lines)/batch_size),
                    epochs=8,
                    verbose=1)
                   

model.save('model.h5')


    
   