# -*- coding: utf-8 -*-
"""
Created on Mon May  4 22:48:02 2020

@author: johan
"""

import os
from utils import file_utils
from PIL import Image
import numpy as np
from dataclasses import dataclass, field
from typing import Any
from threading import Event
from utils import eval_utils
import pickle
import re
import extract_stats
from yolo3.utils import letterbox_image


@dataclass(order=True)
class PrioritizedItem:
    priority: int
    item: Any=field(compare=False)


def detect_colors(working_dir,frame_queue,labels,progress_callback=None, pause_event=None):
    
    if pause_event != None and pause_event.is_set():
        return

    

    detected_numbers_path = os.path.join(working_dir,"detected_numbers")
    os.makedirs(detected_numbers_path,exist_ok=True)
    
    
    detected_bees_folder = os.path.join(working_dir,"detected_bees")
    bee_image_paths = file_utils.get_all_image_paths_in_folder(detected_bees_folder)
    
    

    detected_colors_path=os.path.join(working_dir,"detected_colors.pkl")

    if not os.path.isfile(detected_colors_path):
        
        
        detected_colors = {}

        #Reload checkpoint if existing
        detected_colors_partial_path=os.path.join(working_dir,"detected_colors_partial.pkl")
        if os.path.isfile(detected_colors_partial_path):
            with open(detected_colors_partial_path, 'rb') as f:
                detected_colors = pickle.load(f)        
        else:
            #Otherwise load the detected_bees map
            detected_bees_file = os.path.join(working_dir,"detected_bees.pkl")
            with open(detected_bees_file, 'rb') as f:
                detected_colors = pickle.load(f)

        
        extract_stats.enumerate_bee_detections(detected_colors,working_dir,image_size=(3840,2160))
        
        extract_stats.apply_holes_to_bee_detections(detected_colors,working_dir,image_size=(3840,2160))
    
        progress_callback("Starting to detect colors: " + working_dir,working_dir)
    
    
        for index,bee_image_path in enumerate(bee_image_paths):
            
            frame_number = int(re.search('frame(.*)_detection', bee_image_path).group(1))
            detection_number = int(re.search('_detection(.*).png', bee_image_path).group(1))
            
            if "color" in detected_colors[frame_number][detection_number]:
                continue
                
            
            if index % 100 == 0:
                progress_callback(index/len(bee_image_paths),working_dir)
                with open(detected_colors_partial_path, 'wb') as f:
                    pickle.dump(detected_colors,f)
                if pause_event != None and pause_event.is_set():
                    return
            
            if detected_colors[frame_number][detection_number]["id"] == -1:
                continue

            if detected_colors[frame_number][detection_number]["start"] == None and detected_colors[frame_number][detection_number]["end"] == None:
                continue
            
            if detected_colors[frame_number][detection_number]["start"] == detected_colors[frame_number][detection_number]["end"]:
                continue

            
            
            image = Image.open(bee_image_path)
            detections = get_detections_yolo(image,frame_queue,1,labels)
            
            
            if len(detections) == 0:
                detected_colors[frame_number][detection_number]["color"] = None
                detected_colors[frame_number][detection_number]["color_score"] = 0.0
                continue
            
            best_detection = detections[0]
            for detection in detections:
                if detection["score"] > best_detection["score"]:
                    best_detection = detection
            
    
            number_save_path = os.path.join(detected_numbers_path,os.path.basename(bee_image_path))
            [top,left,bottom,right] = best_detection["bounding_box"]
            width, height = image.size
            top = int(top*height)
            bottom = int(bottom*height)
            left = int(left*width)
            right = int(right*width)
            cropped_image = image.crop((left, top, right, bottom))
            cropped_image.save(number_save_path,"PNG")
            detected_colors[frame_number][detection_number]["color"] = best_detection["name"]
            detected_colors[frame_number][detection_number]["color_score"] = best_detection["score"]
                
    
        with open(detected_colors_path, 'wb') as f:
            pickle.dump(detected_colors,f)
    
    progress_callback(1.0,working_dir)
    progress_callback("Finished detecting colors",working_dir)

        

def get_detections_yolo(image,queue,priority,labels,min_confidence_score = 0.3):
    
    model_image_size=(320,320)
    boxed_image = letterbox_image(image, tuple(reversed(model_image_size)))
    image_data = np.array(boxed_image, dtype='float32')
    image_data /= 255.
    image_expand = np.expand_dims(image_data, 0)  # Add batch dimension.
    
    is_done = Event()
    detections_dict = {}
    queue_item = PrioritizedItem(1/priority,(image_expand,image.size,detections_dict,is_done))
    queue.put(queue_item)
    is_done.wait()

    detections = []
    width,height=image.size
    for box,score,clazz in zip(detections_dict["detection_boxes"],detections_dict["detection_scores"],detections_dict["detection_classes"]):
        top = box[0]/height
        left = box[1]/width
        bottom = box[2]/height
        right = box[3]/width
        detections.append({"bounding_box": [top,left,bottom,right], "score": float(score), "name": labels[clazz]})
    
    detections = eval_utils.non_max_suppression(detections,0.5)
    return detections

    
        
def get_detections(image,queue,priority,labels,min_confidence_score = 0.5):
    image_np = np.asarray(image)         
    image_expand = np.expand_dims(image_np, 0)
    
    is_done = Event()
    detections_dict = {}
    queue_item = PrioritizedItem(1/priority,(image_expand,detections_dict,is_done))
    queue.put(queue_item)
    is_done.wait()
    
    
    detections = []
    count = 0
    for i,score in enumerate(detections_dict['detection_scores']):
        if score >= min_confidence_score:
            count += 1
            top = detections_dict['detection_boxes'][i][0]
            left = detections_dict['detection_boxes'][i][1]
            bottom = detections_dict['detection_boxes'][i][2]
            right = detections_dict['detection_boxes'][i][3]
            detection_class = detections_dict['detection_classes'][i]
            detection_name = labels[detection_class-1] 
            detections.append({"bounding_box": [top,left,bottom,right], "score": float(score), "name": detection_name})
    
    detections = eval_utils.non_max_suppression(detections,0.5)
    
    return detections
