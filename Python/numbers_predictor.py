# -*- coding: utf-8 -*-
"""
Created on Tue May  5 11:22:33 2020

@author: johan
"""

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import keras
from keras import models
from keras import layers
from utils import file_utils
from PIL import Image
import pickle
import re
import os

image_size=32

def start(trained_model,working_dirs,stop_event,progress_callback):
    
    
    model = get_model(trained_model)


    for working_dir in working_dirs:
        if stop_event.is_set():
            return
        
        if os.path.isfile(os.path.join(working_dir,"detected_numbers.pkl")):
            continue

        progress_callback("Starting to detect numbers: ", working_dir)

        detection_map = {}
        with open(os.path.join(working_dir,"detected_colors.pkl"), 'rb') as f:
            detection_map = pickle.load(f)

        detected_numbers_path = os.path.join(working_dir,"detected_numbers")
        all_images = file_utils.get_all_image_paths_in_folder(detected_numbers_path)

        X_matrix = get_data(detected_numbers_path)
        results = model.predict(X_matrix)
        
        
        scores = np.max(results,axis=1)
        labels = np.argmax(results, axis = 1)
        
        for score, label,image_path in zip(scores,labels,all_images):
            frame_number = int(re.search('frame(.*)_detection', image_path).group(1))
            detection_number = int(re.search('_detection(.*).png', image_path).group(1))
            detection_map[frame_number][detection_number]["number"] = str(label)
            detection_map[frame_number][detection_number]["number_score"] = score

        
        with open(os.path.join(working_dir,"detected_numbers.pkl"), 'wb') as f:
            pickle.dump(detection_map,f)
        progress_callback(1.0,working_dir)
        progress_callback("Done detecting numbers: ", working_dir)




def get_data(folder,rgb=True):
    
    all_images = file_utils.get_all_image_paths_in_folder(folder)
    num_images = len(all_images)
    if rgb:
        X = np.empty((num_images,image_size,image_size,3))
    else:
        X = np.empty((num_images,image_size,image_size,1))
    print(num_images)
    for index,image_path in enumerate(all_images):
        image = Image.open(image_path)
        image = image.resize((image_size,image_size))
        if rgb:
            np_image = np.asarray(image).reshape(image_size,image_size,3)
        else:
            image = image.convert('L')
            np_image = np.asarray(image).reshape(image_size,image_size,1)
        X[index] = np_image
    return X

    

def get_model(trained_model,input_shape=(image_size,image_size,3), num_classes = 9):
    
    
    
    
    num_classes = 9
    
    
    keras.backend.clear_session()
    
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), padding='same', 
                               activation='relu',
                               input_shape=(image_size, image_size, 3)),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(32, (3, 3), padding='same', 
                            activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.3),
        
        keras.layers.Conv2D(64, (3, 3), padding='same', 
                               activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(64, (3, 3), padding='same',
                            activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.3),
        keras.layers.Conv2D(128, (3, 3), padding='same', 
                               activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(128, (3, 3), padding='same',
                            activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.3),
        
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.4),    
        keras.layers.Dense(num_classes,  activation='softmax')
    ])
    
    optimizer = keras.optimizers.Adam(lr=1e-3, amsgrad=True)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])
    model.load_weights(trained_model)

    return model

    
    
    
    
    '''
    model = models.Sequential()
    # add Convolutional layers
    model.add(layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same',
                     input_shape=input_shape))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))    
    model.add(layers.Flatten())
    # Densely connected layers
    model.add(layers.Dense(128, activation='relu'))
    # output layer
    model.add(layers.Dense(num_classes, activation='softmax'))
    # compile with adam optimizer & categorical_crossentropy loss function
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    model.load_weights(trained_model)
    return model
    
    
    
    
    
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = input_shape))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Dropout(0.5))
    model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Dropout(0.5))
    model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation = 'relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation = 'softmax'))
        
    model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['acc'])
    
    model.load_weights(trained_model)
    return model
    '''
