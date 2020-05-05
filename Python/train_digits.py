# -*- coding: utf-8 -*-
"""
Created on Mon May  4 20:28:23 2020

@author: johan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn.model_selection import  train_test_split
from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical
from utils import file_utils
from utils import constants
from PIL import Image
from keras.callbacks import ModelCheckpoint

import os


def get_data(folder):
    all_images = file_utils.get_all_image_paths_in_folder(folder)
    num_images = len(all_images)
    X = np.empty((num_images,28,28,1))
    y = np.empty((num_images,))
    for index,image_path in enumerate(all_images):
        image = Image.open(image_path)
        image = image.resize((28,28))
        image = image.convert('L')
        np_image = np.asarray(image).reshape(28,28,1)/255.0
        X[index] = np_image
        annotations = file_utils.get_annotations(image_path)
        y[index] = int(annotations[0]["name"])-1
    y = to_categorical(y, num_classes = 8)
    return X,y

def train(project_dir):
    
    print("Loading data...")
    training_data_dir=os.path.join(project_dir,"images/train")
    #X_train,y_train = get_data(training_data_dir)
    validation_data_dir=os.path.join(project_dir,"images/validation")
    X_val,y_val = get_data(validation_data_dir)
    test_data_dir=os.path.join(project_dir,"images/test")
    X_test,y_test = get_data(test_data_dir)

    
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (28, 28, 1)))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Dropout(0.5))
    model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Dropout(0.5))
    model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation = 'relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(8, activation = 'softmax'))
    
    model.summary()
    
    model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['acc'])
    
    
    model_save_path=os.path.join(project_dir,"trained_model.h5")
    mc = ModelCheckpoint(mode='max', filepath=model_save_path, monitor='val_acc', save_best_only='True', save_weights_only='True', verbose=1)

    model.load_weights(model_save_path)

    #history = model.fit(X_train, y_train, epochs = 400, batch_size = 2048, validation_data = (X_val, y_val),callbacks=[mc])
    score = model.evaluate(X_test, y_test, verbose=0)
    print(score)
    results = model.predict(X_test)
    
    scores = np.max(results,axis=1)
    labels = np.argmax(results, axis = 1) + 1 
    for score, label, in zip(scores,labels):
        print(score)
        print(label)
    
    print(results)



if __name__== "__main__":
    train(constants.project_folder)