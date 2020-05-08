# -*- coding: utf-8 -*-
"""
Created on Fri May  8 14:20:00 2020

@author: johan
"""

import os
import pickle


MAX_SQUARED_DISTANCE = 0.1
min_consecutive_frames_to_be_counted = 3

image_size=(640,360)





def extract_stats(working_dirs):
    
    for working_dir in working_dirs:
        
        detection_map = {}
        detection_map_file = os.path.join(working_dir,"detected_numbers.pkl")
        with open(detection_map_file, 'rb') as f:
            detection_map = pickle.load(f)
        
        enumerate_bee_detections(detection_map,image_size)
        
        apply_holes_to_bee_detections(detection_map,working_dir)
        
        filter_numbers_and_colors(detection_map)
        
        with open(os.path.join(working_dir,"combined_detections.pkl"), 'wb') as f:
            pickle.dump(detection_map,f)

        
        
        
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
        return sorted_by_score[0]
                
    for bee_id in id2color.keys():
        id2color[bee_id] = find_best_prediction(id2color[bee_id])
    
    for bee_id in id2number.keys():
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
      
        
        
        
        
        
        
        
        
        
        
        
        
        
        
'''
Add start and end point to each bee flight using the detected holes
'''      
        
        
        
        
        
        
        
    
def apply_holes_to_bee_detections(detection_map,working_dir):
    
            
    
    #Load hole detections
    detected_holes_path= os.path.join(working_dir,"detected_holes.pkl")
    holes = []
    with open(detected_holes_path, 'rb') as f:
        holes = pickle.load(f)


    
    def is_id_in_frame(bee_id, frame_number):
        if frame_number in detection_map and detection_map[frame_number] != "Skipped":
            for detection in detection_map[frame_number]:
                if detection["id"] == bee_id:
                    return True
        return False
            

    
    starts = {}
    ends = {}
    
    frame_number = 0
    while frame_number in detection_map:
        
        if detection_map[frame_number] and detection_map[frame_number] != "Skipped":
            for detection in detection_map[frame_number]:
                bee_id = detection["id"]
                [top,left,bottom,right] = detection["bounding_box"]
                (center_x,center_y) = ((right+left)/2,(bottom+top)/2)

                if bee_id == -1:
                    continue
                #check if the bee with bee_id was already present in the previous frame
                if not is_id_in_frame(bee_id,frame_number-1):
                    if detection["name"] == "bee":
                        #Bee is sitting
                        starts[bee_id] = get_hole_id_at_position(center_x,center_y,holes)

                    else:
                        starts[bee_id] = None
                if not is_id_in_frame(bee_id,frame_number+1):
                    if detection["name"] == "bee":
                        #Bee is sitting
                        ends[bee_id] = get_hole_id_at_position(center_x,center_y,holes)

                    else:
                        ends[bee_id] = None    
        frame_number += 1
    
    frame_number = 0
    while frame_number in detection_map:
        
        if detection_map[frame_number] and detection_map[frame_number] != "Skipped":
            for detection in detection_map[frame_number]:
                bee_id = detection["id"]
                if bee_id == -1:
                    continue
                detection["start"] = starts[bee_id]
                detection["end"] = ends[bee_id]
        frame_number += 1
        
    print("Detected " + str(len(starts.keys())) + " flights: " + os.path.basename(working_dir))
    
    


    
def get_hole_id_at_position(x,y,holes):
    for hole in holes:
        [top,left,bottom,right] = hole["bounding_box"]
        if x<right and x>left and y<bottom and y>top:
            return hole["id"]
    return None    
    
    
    
    
    
        
        
        
        
        
        
'''
Enumerate Bees such that each bee will have an ID
'''
        
        
        
def enumerate_bee_detections(detection_map,image_size=(1,1)):
    
    #TODO: Make sure that bees are kept being tracked even if for one or two frames the object detection algorithm didn't detect them
    
    (width,height) = image_size
    width_height_ratio = width/height
    bee_id_count = {}
    
    current_bee_index = 0
    frame_number = 0
    while frame_number in detection_map:
        
        if frame_number in detection_map and detection_map[frame_number] != "Skipped":
            
            prev_detections = []
            if frame_number > 0 and detection_map[frame_number-1] != "Skipped":
                prev_detections = detection_map[frame_number-1]
                
            curr_detections = detection_map[frame_number]
            for curr_detection in curr_detections:
                curr_detection.pop('id', None)
                curr_detection.pop('final_color', None)
                curr_detection.pop('final_number', None)

            distances = []
            for prev_index,prev_detection in enumerate(prev_detections):
                for curr_index,curr_detection in enumerate(curr_detections):
                    [top,left,bottom,right] = curr_detection["bounding_box"]
                    (x, y) = ((right+left)/2,(bottom+top)/2)
                    [top,left,bottom,right] = prev_detection["bounding_box"]
                    (prev_x, prev_y) = ((right+left)/2,(bottom+top)/2)
                    squared_distance = pow(x-prev_x,2) + pow((y-prev_y)/width_height_ratio,2)
                    distances.append((squared_distance,prev_detection,curr_detection))
            
            distances = sorted(distances, key = lambda tup: tup[0]) 
            
            already_assigned_prevs = []
            for distance,prev_detection,curr_detection in distances:
                if distance < MAX_SQUARED_DISTANCE and not prev_detection in already_assigned_prevs and not "id" in curr_detection:
                    curr_detection["id"] = prev_detection["id"]
                    already_assigned_prevs.append(prev_detection)
            
            for curr_detection in curr_detections:
                if not "id" in curr_detection:
                    curr_detection["id"] = current_bee_index
                    current_bee_index += 1

                            
            #Updating bee_id_count with detections in current frame
            for detection in detection_map[frame_number]:
                if not detection["id"] in bee_id_count:
                    bee_id_count[detection["id"]] = 1
                else:
                    bee_id_count[detection["id"]] += 1


        frame_number += 1
        
    
        
    frame_number = 0
    #Clean up, remove all detections that are only one frame long  
    while frame_number in detection_map:
        if detection_map[frame_number] and detection_map[frame_number] != "Skipped":
            for detection in detection_map[frame_number]:
                if bee_id_count[detection["id"]] < min_consecutive_frames_to_be_counted:
                    detection["id"] = -1
        frame_number += 1







        
        

    