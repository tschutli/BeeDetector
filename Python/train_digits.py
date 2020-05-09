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
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
import random
import os
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Activation
from keras.layers.normalization import BatchNormalization
import keras
from sklearn.utils import class_weight

image_size=32


def get_data_rgb(folder):
    all_images = file_utils.get_all_image_paths_in_folder(folder)
    num_images = len(all_images)
    X = np.empty((num_images,image_size,image_size,3))
    y = np.empty((num_images,))
    print(len(all_images))
    for index,image_path in enumerate(all_images):
        image = Image.open(image_path)
        image = image.resize((image_size,image_size))
        np_image = np.asarray(image).reshape(image_size,image_size,3)
        X[index] = np_image
        annotations = file_utils.get_annotations(image_path)
        y[index] = int(annotations[0]["name"])-1
    class_weights = class_weight.compute_class_weight('balanced',np.unique(y),y)
    y = to_categorical(y, num_classes = 8)

    return X,y,class_weights


def get_data(folder):
    all_images = file_utils.get_all_image_paths_in_folder(folder)
    num_images = len(all_images)
    X = np.empty((num_images,image_size,image_size,1))
    y = np.empty((num_images,))
    print(len(all_images))
    for index,image_path in enumerate(all_images):
        image = Image.open(image_path)
        image = image.resize((image_size,image_size))
        image = image.convert('L')
        np_image = np.asarray(image).reshape(image_size,image_size,1)
        X[index] = np_image
        annotations = file_utils.get_annotations(image_path)
        y[index] = int(annotations[0]["name"])-1
    class_weights = class_weight.compute_class_weight('balanced',np.unique(y),y)
    y = to_categorical(y, num_classes = 8)
    return X,y,class_weights

def get_predict_data(folder,portion=1.0):
    all_images = file_utils.get_all_image_paths_in_folder(folder)
    num_images = int(np.ceil(len(all_images)*portion))
    X = np.empty((num_images,image_size,image_size,1))
    print(num_images)
    random.shuffle(all_images)
    for index,image_path in enumerate(all_images):
        image = Image.open(image_path)
        image = image.resize((image_size,image_size))
        image = image.convert('L')
        np_image = np.asarray(image).reshape(image_size,image_size,1)
        X[index] = np_image
        if index==num_images-1:
            break
    return X


def get_model_svhn(input_shape,num_classes,learning_rate=0.001):
# Define actual model

    keras.backend.clear_session()
    
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), padding='same', 
                               activation='relu',
                               input_shape=(32, 32, 3)),
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

    return model


def get_model(input_shape,num_classes):
    
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
    return model


def get_old_model(input_shape,num_classes):
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

    return model



def train(project_dir):
    
    print("Loading data...")
    training_data_dir=os.path.join(project_dir,"images/train")
    X_train,y_train,class_weights = get_data_rgb(training_data_dir)
    validation_data_dir=os.path.join(project_dir,"images/validation")
    X_val,y_val,_ = get_data_rgb(validation_data_dir)
    test_data_dir=os.path.join(project_dir,"images/test")
    X_test,y_test,_ = get_data_rgb(test_data_dir)


    model = get_model_svhn((image_size,image_size,3),8)
    
        
    
    model_save_path=os.path.join(project_dir,"trained_model.h5")
    mc = ModelCheckpoint(mode='max', filepath=model_save_path, monitor='val_acc', save_best_only='True', save_weights_only='True', verbose=1)

    #model.load_weights(model_save_path)
    
    batch_size=512
    
    datagen = ImageDataGenerator(rescale=1./255,height_shift_range=5,width_shift_range=5,rotation_range=360,brightness_range=[0.6,1.4],zoom_range=0.1)
    it = datagen.flow(X_train, y_train,batch_size=batch_size)
    
    '''
    it = datagen.flow(X_train, y_train,batch_size=1)
    # generate samples and plot
    for i in range(9):
    	# define subplot
        pyplot.subplot(330 + 1 + i)
    	# generate batch of images
        batch = it.next()
    	# convert to unsigned integers for viewing
        image = batch[0].astype('uint8')
        image = image[0][:,:,0]
    	# plot raw pixel data
        pyplot.imshow(image,cmap='gray',vmin=0,vmax=255)
    # show the figure
    pyplot.show()
    '''
    
    history = model.fit_generator(it,
                                  steps_per_epoch=np.ceil(y_train.shape[0]*3/batch_size),
                                  epochs = 1000, 
                                  validation_data = (X_val, y_val),
                                  callbacks=[mc],
                                  class_weight=class_weights)
    
    #history = model.fit(X_train, y_train, epochs = 400, batch_size = 2048, validation_data = (X_val, y_val),callbacks=[mc])
    
    score = model.evaluate(X_test, y_test, verbose=0)
    print(score)
    

def predict_with_visualization(project_dir,input_folder):
    
    
    
    model_save_path=os.path.join(project_dir,"trained_model.h5")
    model = get_old_model((image_size,image_size,1),8)
    model.load_weights(model_save_path)
    
    print("Evaluating Test Data")
    test_data_dir=os.path.join(project_dir,"images/test")
    X_test,y_test = get_data(test_data_dir)
    results = model.evaluate(X_test,y_test)
    print(results)
    
    print("Making predictions...")
    results = model.predict(X_test)
    y_pred = np.argmax(results, axis = 1)+1
    y_test = np.argmax(y_test, axis = 1)+1
    conf_matrix = confusion_matrix(y_test, y_pred,labels=[1,2,3,4,5,6,7,8],normalize="true")*100
    np.set_printoptions(precision=1,suppress=True)
    
    pd.set_option('display.max_columns', None)

    df = pd.DataFrame(conf_matrix, columns=[1,2,3,4,5,6,7,8], index=[1,2,3,4,5,6,7,8])
    df = df.round(1)
    print(df)


    X_test = get_predict_data(input_folder,portion=0.01)
    results = model.predict(X_test)
    scores = np.max(results,axis=1)
    labels = np.argmax(results, axis = 1) + 1 

    np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
    for i in range(50):
    	# define subplot
        #pyplot.subplot(20,1,i+1)
    	# generate batch of images
        image = X_test[i][:,:,0].astype('uint8')
    
        pyplot.imshow(image,cmap='gray',vmin=0,vmax=255)
    # show the figure
        pyplot.show()
        print(labels[i])
        print(scores[i])
        print(results[i])
    
    

if __name__== "__main__":
    #predict_with_visualization(constants.project_folder,"C:/Users/johan/Desktop/analysis/Test4_0.mp4/detected_numbers")
    train(constants.project_folder)