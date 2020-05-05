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
from utils import file_utils
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


def start(video_path, queue, working_dir, image_size=(640,480),progress_callback=None, pause_event=None):
    
    if pause_event != None and pause_event.is_set():
        return

    detection_map = {}
    
    
    #Reload checkpoint if existing
    frame_number_before_interruption = 0
    detection_map_file = os.path.join(working_dir,"detection_map.pkl")
    checkpoint_file = os.path.join(working_dir,"interruption_checkpoint.pkl")
    if os.path.isfile(checkpoint_file):
        with open(checkpoint_file, 'rb') as f:
            frame_number_before_interruption = pickle.load(f)
    elif os.path.isfile(detection_map_file):
        progress_callback(1.0,video_path)
        return
    
    progress_callback("Starting to detect bees", video_path)

    
    last_24_frames = [None] * 24
    skip_counter = -1

    frames_without_bees_saved = 0
    
    detected_bees_store_path = os.path.join(working_dir,"detected_bees")
    frames_without_bees_path = os.path.join(working_dir,"frames_without_bees")
    os.makedirs(detected_bees_store_path,exist_ok=True)
    os.makedirs(frames_without_bees_path,exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    no_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    #print("Number of frames: " + str(no_of_frames))
    #start_frame = int(no_of_frames / num_processes)* thread_id
    #end_frame = int(no_of_frames / num_processes)* (thread_id+1)
    #cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    #for frame_number in range(start_frame,end_frame):
    for frame_number in range(0,no_of_frames):
        

        
        if (frame_number) % 100 == 0:
                        
            progress_callback(frame_number/no_of_frames,video_path)
            
            if pause_event != None and pause_event.is_set():
                with open(os.path.join(working_dir,"detection_map.pkl"), 'wb') as f:
                    pickle.dump(detection_map,f)
                with open(checkpoint_file, 'wb') as f:
                    pickle.dump(frame_number,f)

                cap.release()
                return


        
        ret, original_image = cap.read()
        if not ret:
            print("BREAK")
            break
        
        last_24_frames[frame_number%24] = original_image

        if frame_number<frame_number_before_interruption:
            continue
        
        if skip_counter>=0 and skip_counter < 12:
            skip_counter += 1
            detection_map[frame_number] = "Skipped"
            continue                    


        resized_image = cv2.resize(original_image, image_size)
        
        detections = get_detections(resized_image,queue,1)
        
        '''
        if len(detections) > 0:
            #Save detection results, including image to folder (so that it can be edited in labelimg later.)
            annotation_folder = os.path.join(working_dir,"bee_annotations")
            os.makedirs(annotation_folder,exist_ok=True)
            minutes = str(int(frame_number/fps/60)).zfill(3)
            seconds = str(int(frame_number/fps) % 60).zfill(2)
            frame = str(frame_number % fps).zfill(2)
            
            
            annotation_im_path = os.path.join(annotation_folder,"min_" + minutes+ "_sec_" + seconds + "_frame_" + frame + ".jpg")
            cv2.imwrite(annotation_im_path,original_image)
            annotation_xml_path = os.path.join(annotation_folder,"min_" + minutes+ "_sec_" + seconds + "_frame_" + frame + ".xml")
            file_utils.save_annotations_to_xml(detections, annotation_im_path, annotation_xml_path)
        '''
        
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
            
    #TODO remove this
    with open(os.path.join(working_dir,"detection_map_before_enumerate.pkl"), 'wb') as f:
        pickle.dump(detection_map,f)

    
    enumerate_detections(detection_map)
    
    with open(os.path.join(working_dir,"detection_map.pkl"), 'wb') as f:
        pickle.dump(detection_map,f)
    if os.path.isfile(checkpoint_file):
        os.remove(checkpoint_file)
    # Release resources
    cap.release()
    progress_callback(1.0,video_path)
    progress_callback("Finished detecting bees", video_path)

    



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
                    detection_map[frame_number][assigned_ids.index(detection["id"])]["id"] = current_bee_index
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
            #TODO: Make sure class 2 is always bee
            if detection_class == 2:
                detection_name = "bee"
            else:
                detection_name = "bee flying"
            detections.append({"bounding_box": [top,left,bottom,right], "score": float(score), "name": detection_name})
    
    detections = eval_utils.non_max_suppression(detections,0.5)
    
    return detections

