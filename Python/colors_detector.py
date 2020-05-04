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

@dataclass(order=True)
class PrioritizedItem:
    priority: int
    item: Any=field(compare=False)


def detect_colors(working_dir,frame_queue,labels,progress_callback=None, pause_event=None):
    
    detected_bees_folder = os.path.join(working_dir,"detected_bees")
    bee_image_paths = file_utils.get_all_image_paths_in_folder(detected_bees_folder)
    
    
    for index,bee_image_path in enumerate(bee_image_paths):
        
        if index % 100 == 0:
            progress_callback(index/len(bee_image_path),working_dir)
            
        if pause_event != None and pause_event.is_set():
            return


        image = Image.open(bee_image_path)
        detections = get_detections(image,frame_queue,1,labels)
        
        if len(detections) > 0:
            

        
        
        
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
