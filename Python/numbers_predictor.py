# -*- coding: utf-8 -*-
"""
Created on Tue May  5 11:22:33 2020

@author: johan
"""

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from keras import models
from keras import layers
from utils import file_utils
from PIL import Image
import pickle
import re
import os


def start(trained_model,working_dirs,stop_event,progress_callback):
    
    model = get_model(trained_model)


    for working_dir in working_dirs:
        if stop_event.is_set():
            return
        progress_callback("Starting to detect numbers: ", working_dir)

        detection_map = {}
        with open(os.path.join(working_dir,"detection_map.pkl"), 'rb') as f:
            detection_map = pickle.load(f)

        detected_numbers_path = os.path.join(working_dir,"detected_numbers")
        all_images = file_utils.get_all_image_paths_in_folder(detected_numbers_path)

        X_matrix = get_data(detected_numbers_path)
        results = model.predict(X_matrix)
        
        
        scores = np.max(results,axis=1)
        labels = np.argmax(results, axis = 1) + 1 
        
        for score, label,image_path in zip(scores,labels,all_images):
            frame_number = int(re.search('frame(.*)_detection', image_path).group(1))
            detection_number = int(re.search('_detection(.*).png', image_path).group(1))
            detection_map[frame_number][detection_number]["number"] = str(label)
            detection_map[frame_number][detection_number]["number_score"] = score


        filter_numbers_and_colors(detection_map)
        
        with open(os.path.join(working_dir,"detection_map.pkl"), 'wb') as f:
            pickle.dump(detection_map,f)
        progress_callback(1.0,working_dir)
        progress_callback("Done detecting numbers: ", working_dir)



def filter_numbers_and_colors(detection_map):
    
    id2color = {}
    id2number = {}
    frame_number = 0
    while frame_number in detection_map:
        if detection_map[frame_number] and detection_map[frame_number] != "Skipped":
            for detection in detection_map[frame_number]:
                if "color" in detection and detection["color"] != None:
                    bee_id = detection["id"]
                    if not bee_id in id2color:
                        id2color[bee_id] = []
                        id2number[bee_id] = []
                    id2color[bee_id].append((detection["color"],detection["color_score"]))
                    id2number[bee_id].append((detection["number"],detection["number_score"]))
        frame_number += 1
    
    def find_best_prediction(tuple_list):
        sorted_by_score = sorted(tuple_list, key=lambda tup: tup[1],reverse=True)
        print(sorted_by_score)
        return sorted_by_score[0]
                
    for bee_id in id2color.keys():
        id2color[bee_id] = find_best_prediction(id2color[bee_id])
    
    for bee_id in id2number.keys():
        print(bee_id)
        id2number[bee_id] = find_best_prediction(id2number[bee_id])

        
    frame_number = 0
    while frame_number in detection_map:
        if detection_map[frame_number] and detection_map[frame_number] != "Skipped":
            for detection in detection_map[frame_number]:
                bee_id = detection["id"]
                if bee_id in id2color:
                    detection["final_color"] = id2color[bee_id][0]
                    detection["final_number"] = id2number[bee_id][0]
                             
        frame_number += 1

def get_data(folder):
    all_images = file_utils.get_all_image_paths_in_folder(folder)
    num_images = len(all_images)
    X = np.empty((num_images,28,28,1))
    for index,image_path in enumerate(all_images):
        image = Image.open(image_path)
        image = image.resize((28,28))
        image = image.convert('L')
        np_image = np.asarray(image).reshape(28,28,1)/255.0
        X[index] = np_image
    return X


def get_model(trained_model):
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
        
    model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['acc'])
    
    model.load_weights(trained_model)
    return model