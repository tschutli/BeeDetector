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


def start(thread_id, num_processes, video_path, queue, working_dir, image_size=(640,480)):
    
    
    detection_map = {}
    last_24_frames = [None] * 24
    skip_counter = -1

    
    
    cap = cv2.VideoCapture(video_path)
    no_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  
    start_frame = int(no_of_frames / num_processes)* thread_id
    end_frame = int(no_of_frames / num_processes)* (thread_id+1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    for frame_number in range(start_frame,end_frame):
        
        if (frame_number-start_frame) % 100 == 0:
            print("Thread " + str(thread_id) + " progress: " + str(frame_number-start_frame) + " / " + str(end_frame-start_frame))

        
        
        ret, original_image = cap.read()
        if not ret:
            break
        
        last_24_frames[frame_number%24] = original_image

        
        if skip_counter>=0 and skip_counter < 12:
            skip_counter += 1
            detection_map[frame_number] = "Skipped"
            continue                    


        resized_image = cv2.resize(original_image, image_size)
        
        detections = get_detections(resized_image,frame_number-start_frame,queue)
        
        
        
        crop_out_detections(original_image,frame_number,detections,working_dir)

        
        if not detections:
            skip_counter = 0
        elif skip_counter == 12:
            skip_counter = -1
            
            for i in range(1,13):
                if frame_number-i < start_frame:
                    break
                original_image = last_24_frames[(frame_number-i)%24]
                resized_image = cv2.resize(original_image, image_size)
                detections = get_detections(resized_image,frame_number-i-start_frame,queue)
                detection_map[frame_number-i] = detections
                crop_out_detections(original_image,frame_number-i,detections,working_dir)
                if not detections:
                    break
                       
        else:
            skip_counter = -1
            

            #print("qsize: " + str(queue.qsize()))

            
            

    # Release resources
    cap.release()
    print("Thread " + str(thread_id) + ": DONE!")

    
    return detection_map











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

    queue.put((priority, (image_expand,detections_dict,is_done)))
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

