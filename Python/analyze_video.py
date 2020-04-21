# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 12:52:15 2020

@author: johan
"""

from utils import constants
import cv2
import progressbar
import time
import os
import pickle
import queue
import threading
import frame_reader
import predictor
from threading import Event
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import numpy as np
from dataclasses import dataclass, field
from typing import Any
from utils import eval_utils
from utils import file_utils
import string


@dataclass(order=True)
class PrioritizedItem:
    priority: int
    item: Any=field(compare=False)


current_milli_time = lambda: int(round(time.time() * 1000))



image_size = constants.tensorflow_tile_size
image_size = (1000,750)
min_confidence_score = 0.5
num_threads = 7


def analyze_videos(trained_bee_model, trained_hole_model, input_videos, working_dir):
    
    working_dirs = []
    for input_video in input_videos:
        video_dir = os.path.join(working_dir, os.path.basename(input_video))
        while video_dir in working_dirs:
            video_dir += "2"
        working_dirs.append(video_dir)
        os.makedirs(video_dir,exist_ok=True)
          
    detect_bees(trained_bee_model,input_videos,working_dirs)
    detect_holes(trained_hole_model,input_videos,working_dirs)

    #TODO detect numbers on bees
    #TODO get statistics
    
    visualize_videos(input_videos,working_dirs)
    

    
def detect_holes(trained_hole_model, input_videos, working_dirs):
    
    frame_queue = queue.PriorityQueue()
        
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for (input_video,working_dir) in zip(input_videos,working_dirs):
            #if os.path.isfile(os.path.join(working_dir,"detected_holes.pkl")):
            #    continue
            future = executor.submit(hole_frame_reader,working_dir,frame_queue,image_size)
            
            futures.append(future)
        
        
        predictor_stop_event = Event()
        predictor_thread = threading.Thread(target=predictor.start,args=(trained_hole_model,frame_queue,image_size,predictor_stop_event,))
        predictor_thread.daemon=True
        predictor_thread.start()
        
        for future in futures:
            future.result()
        predictor_stop_event.set()


def hole_frame_reader(working_dir,frame_queue,image_size):
    '''
    hole_images_folder = os.path.join(working_dir,"frames_without_bees")
    all_images = file_utils.get_all_image_paths_in_folder(hole_images_folder)
    
    all_detections = []
    num_holes_detected = []
    for image_path in all_images:
        
        image = Image.open(image_path)
        resized_image = image.resize(image_size)
        image_np = np.asarray(resized_image)         
        image_expand = np.expand_dims(image_np, 0)
        is_done = Event()
        detections_dict = {}
        
        queue_item = PrioritizedItem(1,(image_expand,detections_dict,is_done))
        frame_queue.put(queue_item)
        is_done.wait()
        
        detections = []
        for i,score in enumerate(detections_dict['detection_scores']):
            if score >= min_confidence_score:
                top = detections_dict['detection_boxes'][i][0]
                left = detections_dict['detection_boxes'][i][1]
                bottom = detections_dict['detection_boxes'][i][2]
                right = detections_dict['detection_boxes'][i][3]
                detection_class = detections_dict['detection_classes'][i]
                detections.append({"bounding_box": [top,left,bottom,right], "score": float(score), "class": detection_class})
        
        detections = eval_utils.non_max_suppression(detections,0.5)
        all_detections.append(detections)
        num_holes_detected.append(len(detections))

    most_frequent_answer = max(set(num_holes_detected), key = num_holes_detected.count)
    index_of_most_frequent_answer = num_holes_detected.index(most_frequent_answer)
    
    print(num_holes_detected)
    
    holes = all_detections[index_of_most_frequent_answer]
    '''
    holes = []
    
    with open(os.path.join(working_dir,"detected_holes.pkl"), 'rb') as f:
        holes = pickle.load(f)

    enumerate_holes(holes)
    


    detection_map = {}
    with open(os.path.join(working_dir,"detection_map.pkl"), 'rb') as f:
        detection_map = pickle.load(f)

    def is_id_in_frame(bee_id, frame_number):
        if detection_map[frame_number] and detection_map[frame_number] != "Skipped":
            for detection in detection_map[frame_number]:
                if detection["id"] == bee_id:
                    return True
        return False
            

    
    frame_number = 0
    while frame_number in detection_map:
        
        if detection_map[frame_number] and detection_map[frame_number] != "Skipped":
            for detection in detection_map[frame_number]:
                bee_id = detection["id"]
                if bee_id == -1:
                    continue
                if not is_id_in_frame(bee_id,frame_number-1):
                    if detection["class"] == 1:
                        #Bee is sitting
                        print()
        frame_number += 1
    
    '''
    with open(os.path.join(working_dir,"detected_holes.pkl"), 'wb') as f:
        pickle.dump(holes,f)
    '''
    
    
def save_holes_predictions_image(holes,image_path,destination_path):
    
    image = cv2.imread(image_path)
    height, width, channels = image.shape
    
    for hole_id,hole in enumerate(holes):
        [top,left,bottom,right] = hole["bounding_box"]
        top = int(top*height)
        bottom = int(bottom*height)
        left = int(left*width)
        right = int(right*width)
        rectangle_color = (0,0,255)
        
        image = cv2.rectangle(image, (left,top), (right,bottom), rectangle_color, 2)
        show_str = str(hole_id)
        if "name" in hole:
            show_str = hole["name"]
        cv2.putText(image, show_str, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, rectangle_color, 2)
    cv2.imwrite(destination_path,image)


    
def enumerate_holes(detections):
    
    right_neighbours = {}
    
    for hole_id,hole in enumerate(detections):
        [top,left,bottom,right] = hole["bounding_box"]
        (center_x,center_y) = ((right+left)/2,(bottom+top)/2)
        right_neighbours[hole_id] = None
        for hole_id2,hole2 in enumerate(detections):
            if hole_id == hole_id2:
                continue
            [top2,left2,bottom2,right2] = hole2["bounding_box"]
            (center_x2,center_y2) = ((right2+left2)/2,(bottom2+top2)/2)
            if center_y2 < bottom and center_y2 > top and center_x2 > center_x:
                #it should be in the same row and right of the current hole
                squared_distance = pow(center_x-center_x2,2) + pow(center_y-center_y2,2)
                if right_neighbours[hole_id] == None or right_neighbours[hole_id]["distance"] > squared_distance:
                    right_neighbours[hole_id] = {"right_neighbour" : hole_id2, "distance": squared_distance}
    
    right_most_holes = []
    
    for hole_id in right_neighbours.keys():
        if right_neighbours[hole_id] == None:
            [top,left,bottom,right] = detections[hole_id]["bounding_box"]
            (center_x,center_y) = ((right+left)/2,(bottom+top)/2)
            right_most_holes.append((hole_id,center_y))
    
    right_most_holes.sort(key=lambda tup: tup[1])  # sorts in place
    
    print(right_most_holes)
    
    def get_left_most_element_of_row(hole_id):
        left_neighbor = hole_id
        has_found_a_left_neighbor = True
        while has_found_a_left_neighbor:
            has_found_a_left_neighbor = False
            for hole_id2 in right_neighbours.keys():
                if right_neighbours[hole_id2] == None:
                    continue
                #print(str(hole_id2) + "/" + str(right_neighbours[hole_id2]["right_neighbour"]))

                if right_neighbours[hole_id2]["right_neighbour"] == left_neighbor:
                    left_neighbor = hole_id2
                    has_found_a_left_neighbor = True
                    break
                
        return left_neighbor

    


    for index,(hole_id,center_y) in enumerate(right_most_holes):
        letter = string.ascii_uppercase[index]
        current_element = get_left_most_element_of_row(hole_id)

        hole_column = 0

        while True:
            detections[current_element]["name"] = str(str(letter) + str(hole_column))
            hole_column += 1
            if right_neighbours[current_element] == None:
                break
            else:
                current_element = right_neighbours[current_element]["right_neighbour"]
            
        
                
    
    
            
    
    
    

        
        
    
    
    
    
def detect_bees(trained_bee_model,input_videos,working_dirs):
    
        
    #start = current_milli_time()
    
    frame_queue = queue.PriorityQueue()
        
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for (input_video,working_dir) in zip(input_videos,working_dirs):
            if os.path.isfile(os.path.join(working_dir,"detection_map.pkl")):
                continue
            future = executor.submit(frame_reader.start, input_video, frame_queue, working_dir,image_size)
            futures.append(future)
        
        
        predictor_stop_event = Event()
        predictor_thread = threading.Thread(target=predictor.start,args=(trained_bee_model,frame_queue,image_size,predictor_stop_event,))
        predictor_thread.daemon=True
        predictor_thread.start()
        
        for future in futures:
            future.result()
        predictor_stop_event.set()
        

    
def visualize_videos(input_videos,working_dirs):
    
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        
        futures = []
        
        for (input_video,working_dir) in zip(input_videos,working_dirs):
            detection_map = {}
            with open(os.path.join(working_dir,"detection_map.pkl"), 'rb') as f:
                detection_map = pickle.load(f)
            
            future = executor.submit(visualize, input_video, detection_map, os.path.join(working_dir,"visualization.MP4"))
            futures.append(future)

        for future in futures:
            future.result()

            
    


def visualize(input_video,detection_map,output_path):
    #start = current_milli_time()
    #cap = cv2.VideoCapture(input_video)
    #cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    print("Visualizing video: " + input_video)

    out = cv2.VideoWriter(output_path,cv2.VideoWriter_fourcc(*'MP4V'), 25, image_size)

    fps = 25
    

    cap = cv2.VideoCapture(input_video)

    no_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


    for frame_number in progressbar.progressbar(range(0,no_of_frames)):
    
        start = current_milli_time()

        ret, image = cap.read()
        if not ret:
            print("BREAK")
            break
        #print("",flush=True)
        
        #print("1: " + str(current_milli_time()-start))

        image = cv2.resize(image, image_size)
        #start = current_milli_time()
        #cv2.imwrite(output_path[:-4] + ".jpg",image)
        
        #print("2: " + str(current_milli_time()-start), flush=True)
        
        if not frame_number in detection_map:
            print("Did not find frame " + str(frame_number) + " in detection_map.")
            continue
        
        if detection_map[frame_number] and detection_map[frame_number] != "Skipped":
            for detection in detection_map[frame_number]:
                if detection["id"] != -1:
                    [top,left,bottom,right] = detection["bounding_box"]
                    top = int(top*image_size[1])
                    bottom = int(bottom*image_size[1])
                    left = int(left*image_size[0])
                    right = int(right*image_size[0])
                    rectangle_color = (0,0,255)
                    if(detection["class"] == 0):
                        rectangle_color = (0,255,0)
                    elif(detection["class"] == 1):
                        rectangle_color = (255,0,0)      
                    
                    image = cv2.rectangle(image, (left,top), (right,bottom), rectangle_color, 2)
                    
                    cv2.putText(image, str(detection["id"]) + " / " + '{0:.2f}'.format(detection["score"]), (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, rectangle_color, 2)
                    

        #print("3: " + str(current_milli_time()-start), flush=True)
        #if detection_map[frame_number] != "Skipped":
        out.write(image)
            #print("4: " + str(current_milli_time()-start), flush=True)
            #print("", flush=True)

    cap.release() 
    out.release()
    

    


if __name__== "__main__":
    
    bee_model_path = constants.bee_model_path
    hole_model_path = constants.hole_model_path
    input_videos = constants.input_videos
    working_dir = constants.working_dir
    analyze_videos(bee_model_path, hole_model_path, input_videos, working_dir)
    
    