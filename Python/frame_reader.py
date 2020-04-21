# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 11:37:02 2020

@author: johan
"""

import cv2
from PIL import Image
import numpy as np
from threading import Event
import os
from utils import eval_utils
from dataclasses import dataclass, field
from typing import Any
import pickle


MAX_SQUARED_DISTANCE = 0.01
min_consecutive_frames_to_be_counted = 3
number_of_images_without_bees_to_save = 23



@dataclass(order=True)
class PrioritizedItem:
    priority: int
    item: Any=field(compare=False)


def start(video_path, queue, working_dir, image_size=(640,480)):
    
    detection_map = {}
    #TODO: Load existing detection_map
    
    last_24_frames = [None] * 24
    skip_counter = -1

    frames_without_bees_saved = 0
    
    detected_bees_store_path = os.path.join(working_dir,"detected_bees")
    frames_without_bees_path = os.path.join(working_dir,"frames_without_bees")
    os.makedirs(detected_bees_store_path,exist_ok=True)
    os.makedirs(frames_without_bees_path,exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    no_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  
    print("Number of frames: " + str(no_of_frames))
    #start_frame = int(no_of_frames / num_processes)* thread_id
    #end_frame = int(no_of_frames / num_processes)* (thread_id+1)
    #cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    #for frame_number in range(start_frame,end_frame):
    for frame_number in range(0,no_of_frames):
        
        if (frame_number) % 100 == 0:
            print(os.path.basename(video_path) + " progress: " + str(frame_number) + " / " + str(no_of_frames))

        cap_frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)
        if cap_frame_number != frame_number:
            print("ERROR")
            print(cap_frame_number)
            print(frame_number)


        
        ret, original_image = cap.read()
        if not ret:
            print("BREAK")
            break
        
        last_24_frames[frame_number%24] = original_image

        
        if skip_counter>=0 and skip_counter < 12:
            skip_counter += 1
            detection_map[frame_number] = "Skipped"
            continue                    


        resized_image = cv2.resize(original_image, image_size)
        
        detections = get_detections(resized_image,queue,1)
        
        detection_map[frame_number] = detections
        
        crop_out_detections(original_image,frame_number,detections,detected_bees_store_path)

        
        if not detections:
            if frames_without_bees_saved < number_of_images_without_bees_to_save:
                save_path = os.path.join(frames_without_bees_path,str(frame_number) + ".jpg")
                cv2.imwrite(save_path,original_image)
                frames_without_bees_saved += 1
            skip_counter = 0
        elif skip_counter == 12:
            skip_counter = -1
            
            for i in range(1,13):
                if frame_number-i < 0:
                    break
                original_image = last_24_frames[(frame_number-i)%24]
                resized_image = cv2.resize(original_image, image_size)
                detections = get_detections(resized_image,queue,1)
                detection_map[frame_number-i] = detections
                crop_out_detections(original_image,frame_number-i,detections,detected_bees_store_path)
                if not detections:
                    break
                       
        else:
            skip_counter = -1
            
            
    
    enumerate_detections(detection_map)
    
    with open(os.path.join(working_dir,"detection_map.pkl"), 'wb') as f:
        pickle.dump(detection_map,f)

    # Release resources
    cap.release()
    print(os.path.basename(video_path) + ": DONE!")

    



def enumerate_detections(detection_map):
    
    #TODO: Make sure that bees are kept being tracked even if for one or two frames the object detection algorithm didn't detect them
    
    bee_id_count = {}
    
    current_bee_index = 0
    frame_number = 0
    while frame_number in detection_map:
        
        if detection_map[frame_number] and detection_map[frame_number] != "Skipped":
            
            prev_detections = []
            if frame_number > 0:
                prev_detections = detection_map[frame_number-1]
                
                
            for detection in detection_map[frame_number]:
                if prev_detections == [] or prev_detections == "Skipped":
                    detection["id"] = current_bee_index
                    current_bee_index += 1
                else:
                    [top,left,bottom,right] = detection["bounding_box"]
                    (x, y) = ((right+left)/2,(bottom+top)/2)
                    
                    distances = []
                    
                    for prev_index,prev_detection in enumerate(prev_detections):
                        [top,left,bottom,right] = prev_detection["bounding_box"]
                        (prev_x, prev_y) = ((right+left)/2,(bottom+top)/2)
                        squared_distance = pow(x-prev_x,2) + pow(y-prev_y,2)
                        distances.append({"distance": squared_distance, "index": prev_index})
                        #print("{:.2e}".format(squared_distance))
                    
                    distances = sorted(distances, key = lambda i: i['distance']) 
                    
                    if distances[0]["distance"] < MAX_SQUARED_DISTANCE and (len(distances) < 2 or distances[1]["distance"] > MAX_SQUARED_DISTANCE):
                        #there was only one bee close in the previous image. Give it the same id!
                        detection["id"] = prev_detections[distances[0]["index"]]["id"]
                    else:
                        detection["id"] = current_bee_index
                        current_bee_index += 1
                    
            #Making sure that one detection in the previous frame did not split into two detections in the current frame
            assigned_ids = []
            for detection in detection_map[frame_number]:
                if not detection["id"] in assigned_ids:
                    assigned_ids.append(detection["id"])
                else:
                    print("Two objects very close to each other causing confusion at frame: " + str(frame_number))
                    detection_map[frame_number][assigned_ids.index(detection["id"])] = current_bee_index
                    assigned_ids.append(detection["id"])
                    detection["id"] = current_bee_index + 1
                    current_bee_index += 2
            
            #Updating bee_id_count with detections in current frame
            for detection in detection_map[frame_number]:
                if not detection["id"] in bee_id_count:
                    bee_id_count[detection["id"]] = 1
                else:
                    bee_id_count[detection["id"]] += 1


        frame_number += 1
        
    
    assigned_ids = {}
        
    frame_number = 0
    #Clean up, remove all detections that are only one frame long  
    while frame_number in detection_map:
        if detection_map[frame_number] and detection_map[frame_number] != "Skipped":
            for detection in detection_map[frame_number]:
                if bee_id_count[detection["id"]] < min_consecutive_frames_to_be_counted:
                    detection["id"] = -1
        frame_number += 1








def crop_out_detections(image, frame_number, detections, output_dir):
    
    #saving cropped out detections:
    for index,detection in enumerate(detections):
        
        bee_tile_path = os.path.join(output_dir,"frame" + str(frame_number) + "_detection" + str(index) + ".png")
        
        height, width = image.shape[:2]

        [top,left,bottom,right] = detection["bounding_box"]
        top = int(top*height)
        bottom = int(bottom*height)
        left = int(left*width)
        right = int(right*width)
        
        bee_tile = image[top:bottom,left:right]
        
        cv2.imwrite(bee_tile_path, bee_tile)


def get_detections(resized_image,queue,priority,min_confidence_score = 0.5):
    image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
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
            detections.append({"bounding_box": [top,left,bottom,right], "score": float(score), "class": detection_class})
    
    detections = eval_utils.non_max_suppression(detections,0.5)
    
    return detections

