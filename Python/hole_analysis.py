# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 15:51:14 2020

@author: johan
"""

import os
import pickle
import string
from dataclasses import dataclass, field
from typing import Any
import cv2
from utils import file_utils
from utils import eval_utils
from PIL import Image
import numpy as np
from threading import Event


min_confidence_score = 0.4

@dataclass(order=True)
class PrioritizedItem:
    priority: int
    item: Any=field(compare=False)


def hole_frame_reader(working_dir,frame_queue,image_size,progress_callback=None, pause_event=None):
    
    if pause_event != None and pause_event.is_set():
        return
    
    hole_images_folder = os.path.join(working_dir,"frames_without_bees")
    all_images = file_utils.get_all_image_paths_in_folder(hole_images_folder)
    
    detected_holes_xml_path = os.path.join(hole_images_folder,"detected_holes.xml")
    #Only detect holes if detected_holes.pkl does not yet exist
    if not os.path.isfile(detected_holes_xml_path):

        progress_callback("Starting to detect holes", working_dir)
    
        
        
        all_detections = []
        num_holes_detected = []
        image_paths = []
        for index,image_path in enumerate(all_images):
            
            progress_callback(index/len(all_images),working_dir)
    
            if pause_event != None and pause_event.is_set():
                return
    
            if os.path.basename(image_path) == "first_frame.jpg" and len(all_images)>1:
                continue
            
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
                    #detection_class = detections_dict['detection_classes'][i]
                    detections.append({"bounding_box": [top,left,bottom,right], "score": float(score), "name": "hole"})
            
            detections = eval_utils.non_max_suppression(detections,0.5)
            all_detections.append(detections)
            num_holes_detected.append(len(detections))
            image_paths.append(image_path)
        most_frequent_answer = max(set(num_holes_detected), key = num_holes_detected.count)
        index_of_most_frequent_answer = num_holes_detected.index(most_frequent_answer)
            
        holes = all_detections[index_of_most_frequent_answer]
    
        print("Detected " + str(len(holes)) + " holes: " + os.path.basename(working_dir),flush=True)
        
            
        src_image = image_paths[index_of_most_frequent_answer]
        detected_holes_image_path=detected_holes_xml_path[:-4] + ".jpg"
        save_holes_predictions_image(holes,src_image,detected_holes_image_path)
        file_utils.save_annotations_to_xml(holes, detected_holes_image_path,  detected_holes_xml_path)


    holes = file_utils.get_annotations_from_xml(detected_holes_xml_path)
    to_relative_coordinates(holes,detected_holes_xml_path[:-4]+".jpg")
    enumerate_holes(holes)

    with open(os.path.join(working_dir,"detected_holes.pkl"), 'wb') as f:
        pickle.dump(holes,f)
        
        
    progress_callback(1.0,working_dir)
    progress_callback("Finished detecting holes",working_dir)




def to_relative_coordinates(holes,image_path):
    image = Image.open(image_path)
    width, height = image.size
    for hole in holes:
        [top,left,bottom,right] = hole["bounding_box"]
        top = top/height
        bottom = bottom/height
        left = left/width
        right = right/width
        hole["bounding_box"] = [top,left,bottom,right]
    

        
    
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
        
        image = cv2.rectangle(image, (left,top), (right,bottom), rectangle_color, 3)
        show_str = str(hole_id)
        if "id" in hole:
            show_str = hole["id"]
        cv2.putText(image, show_str, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, rectangle_color, 3)
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

        hole_column = 1

        while True:
            detections[current_element]["id"] = str(str(letter) + str(hole_column))
            hole_column += 1
            if right_neighbours[current_element] == None:
                break
            else:
                current_element = right_neighbours[current_element]["right_neighbour"]
    
    unspecified_holes = []
    for hole_index, hole in enumerate(detections):
        if not "id" in hole:
            unspecified_holes.append(hole_index)
    
    unspecified_holes.reverse()
    for unspecified_hole in unspecified_holes:
        detections.pop(unspecified_hole)
    if len(unspecified_holes) > 0:
        print("Found " + str(len(unspecified_holes)) + " hole detections, that could not be enumerated.")
            
        

            