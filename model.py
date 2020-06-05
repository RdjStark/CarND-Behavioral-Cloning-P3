
import os
import csv
import cv2
import sklearn
import numpy as np
from random import shuffle

lines = []
with open('../driving_data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

from sklearn.model_selection import train_test_split        
train_lines, validation_lines = train_test_split(lines, test_size = 0.2)
        
def sample_generator(lines, batch_size=32, side_camera_correction=0.2):    
    num_lines = len(lines)
    while 1:        
        shuffle(lines)
        batch_count = num_lines // batch_size
        for offset in range(0, batch_count, 1):
            images = []
            measurements = []
            sample = lines[offset * batch_size:(offset+1) * batch_size]
            for item in sample:    
                center_image_name = '../driving_data/IMG/' + item[0].split('/')[-1]
                center_image = cv2.imread(center_image_name)   
                images.append(center_image)
                measurements.append(np.float(item[3]))                
                
                left_image_name = '../driving_data/IMG/' + item[1].split('/')[-1]
                left_image = cv2.imread(left_image_name)   
                images.append(left_image)
                measurements.append(np.float(item[3]) + side_camera_correction)                
                
                right_image_name = '../driving_data/IMG/' + item[2].split('/')[-1]
                right_image = cv2.imread(right_image_name)   
                images.append(right_image)
                measurements.append(np.float(item[3]) - side_camera_correction)                
                
            yield(np.array(images), np.array(measurements))
                         
sampler = sample_generator(lines,2)

(imas, meas) = next(sampler)


    
   