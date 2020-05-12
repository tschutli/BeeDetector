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

image_size=50
batch_size=128
rgb=True


def get_data(folder,rgb=True,image_size=image_size,validation_split=0.0):
    
    all_files=[]
    all_labels=[]
    for label in next(os.walk(folder))[1]:
        sub_folder = os.path.join(folder,label)
        for file in os.listdir(sub_folder):
            all_files.append(os.path.join(sub_folder,file))
            all_labels.append(label)
    
    shuffle_list = list(zip(all_files, all_labels))
    random.shuffle(shuffle_list)
    all_files, all_labels = zip(*shuffle_list)

    num_images = len(all_files)
    if rgb:
        X = np.empty((num_images,image_size,image_size,3))
    else:
        X = np.empty((num_images,image_size,image_size,1))
    y = np.empty((num_images,))

        
    for index,image_path in enumerate(all_files):
        image = Image.open(image_path)
        image = image.resize((image_size,image_size))
        if rgb:
            np_image = np.asarray(image).reshape(image_size,image_size,3)
        else:
            image = image.convert('L')
            np_image = np.asarray(image).reshape(image_size,image_size,1)
        X[index] = np_image
        y[index] = int(all_labels[index])
    
    X_val = X[:int(num_images*validation_split)]
    y_val = y[:int(num_images*validation_split)]
    X=X[int(num_images*validation_split):]
    y=y[int(num_images*validation_split):]
    class_weights = class_weight.compute_class_weight('balanced',np.unique(y),y)
    y = to_categorical(y)
    y_val = to_categorical(y_val)
    
    return X,y,X_val,y_val,class_weights



def get_predict_data(folder,portion=1.0,rgb=True):
    all_images = file_utils.get_all_image_paths_in_folder(folder)
    num_images = int(np.ceil(len(all_images)*portion))
    if rgb:
        X = np.empty((num_images,image_size,image_size,3))
    else:
        X = np.empty((num_images,image_size,image_size,1))
    print(num_images)
    random.shuffle(all_images)
    for index,image_path in enumerate(all_images):
        image = Image.open(image_path)
        image = image.resize((image_size,image_size))
        if rgb:
            np_image = np.asarray(image).reshape(image_size,image_size,3)
        else:
            image = image.convert('L')
            np_image = np.asarray(image).reshape(image_size,image_size,1)
        X[index] = np_image
        if index==num_images-1:
            break
    return X


def svhn_layer(model, filters, strides, name,input_shape=None):

    if input_shape:
        model.add(Conv2D(filters, (5, 5), 
               padding='same', name='conv2d_' + name,input_shape=input_shape))
    else:
        model.add(Conv2D(filters, (5, 5), 
           padding='same', name='conv2d_' + name))


    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(strides, strides),
              name='maxpool_2d_' + name))

    model.add(Dropout(0.2))

    return model


def get_model_svhn_2(input_shape,num_classes):
    
    keras.backend.clear_session()

    model = Sequential()
    
    svhn_layer(model, 48, 2, 'hidden1',input_shape=input_shape)
    svhn_layer(model, 48, 1, 'hidden2')
    svhn_layer(model, 48, 1, 'hidden3')
    
    svhn_layer(model, 64, 2, 'hidden4')
    svhn_layer(model, 64, 1, 'hidden5')
    svhn_layer(model, 64, 1, 'hidden6')
    
    svhn_layer(model, 128, 2, 'hidden7')
    svhn_layer(model, 128, 1, 'hidden8')
    svhn_layer(model, 128, 1, 'hidden9')
    
    svhn_layer(model, 192, 2, 'hidden10')
    svhn_layer(model, 192, 1, 'hidden11')
    
    model.add(Flatten())
    
    model.add(Dense(3072))
    model.add(Dense(3072))
    model.add(Dense(num_classes,  activation='softmax'))
        
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

    return model


def get_model_svhn(input_shape,num_classes,learning_rate=0.001):
# Define actual model

    keras.backend.clear_session()
    
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), padding='same', 
                               activation='relu',
                               input_shape=input_shape),
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



def train(data_dir,trained_model):
    
    print("Loading data...")
    X,y,X_val,y_val,class_weights = get_data(data_dir,validation_split=0.1)


    model = get_model_svhn((image_size,image_size,3),9)
    
    model.summary()
    
    mc = ModelCheckpoint(mode='max', filepath=trained_model, monitor='val_acc', save_best_only='True', save_weights_only='True', verbose=1)

    model.load_weights(trained_model)
        
    datagen = ImageDataGenerator(rescale=1./255,height_shift_range=5,width_shift_range=5,rotation_range=360,brightness_range=[0.6,1.4],zoom_range=0.1)
    it = datagen.flow(X, y,batch_size=batch_size)
    
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
                                  steps_per_epoch=np.ceil(y.shape[0]*2/batch_size),
                                  epochs = 1000, 
                                  validation_data = (X_val/255., y_val),
                                  callbacks=[mc],
                                  class_weight=class_weights)
        
    

def predict_with_visualization(project_dir,input_folder):
    
    
    
    model_save_path=os.path.join(project_dir,"trained_model.h5")
    model = get_model_svhn((image_size,image_size,3),9)
    model.load_weights(model_save_path)
    
    '''
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

    '''
    
    X_test = get_predict_data(input_folder,portion=0.01)
    results = model.predict(X_test/255.)
    scores = np.max(results,axis=1)
    labels = np.argmax(results, axis = 1)

    np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
    for i in range(50):
    	# define subplot
        #pyplot.subplot(20,1,i+1)
    	# generate batch of images
        if not rgb:
            image = X_test[i][:,:,0].astype('uint8')
            pyplot.imshow(image,cmap='gray',vmin=0,vmax=255)
        else:
            image = X_test[i].astype('uint8')
            pyplot.imshow(image)


        pyplot.show()
        print(labels[i])
        print(scores[i])
        print(results[i])
    
    

if __name__== "__main__":
    predict_with_visualization(constants.project_folder,"C:/Users/johan/Desktop/analysis/Test4_4.mp4/detected_numbers")
    #train("C:/Users/johan/Desktop/Data/Digit_classification_extended",os.path.join(constants.project_folder,"trained_model.h5"))