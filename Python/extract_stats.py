# -*- coding: utf-8 -*-
"""
Created on Fri May  8 14:20:00 2020

@author: johan
"""

import os
import pickle


min_consecutive_frames_to_be_counted = 3

image_size=(3840,2160)





def extract_stats(working_dirs):
    
    for working_dir in working_dirs:
        
        detection_map = {}
        detection_map_file = os.path.join(working_dir,"detected_numbers.pkl")
        with open(detection_map_file, 'rb') as f:
            detection_map = pickle.load(f)
        
        
        enumerate_bee_detections(detection_map,working_dir,image_size)
        
        apply_holes_to_bee_detections(detection_map,working_dir,image_size=image_size)
        
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
        
        
       
def get_area_of_detection(detection,image_size=(3840,2160)):
    (image_width,image_height) = image_size
    width_height_ratio = image_width/image_height

    [top,left,bottom,right] = detection["bounding_box"]
    width = (right-left)
    height = (bottom-top)/width_height_ratio
    area = width*height
    return area

        
def get_average_hole_area(holes,image_size=(3840,2160)):
    area_sum = 0
    for hole in holes:
        area_sum += get_area_of_detection(hole,image_size=image_size)
    return area_sum/len(holes)
        

    
        
        
        
    
def apply_holes_to_bee_detections(detection_map,working_dir,image_size=(3840,2160)):
    
            
    
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
            

    average_hole_area=get_average_hole_area(holes)
    
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
                        if starts[bee_id] != None and get_area_of_detection(detection) >= average_hole_area:
                            starts[bee_id] = "too large"
                            #print("Detected an erroneous start point: Bee detection is too large.")
                    else:
                        starts[bee_id] = None
                if not is_id_in_frame(bee_id,frame_number+1):
                    if detection["name"] == "bee":
                        #Bee is sitting
                        ends[bee_id] = get_hole_id_at_position(center_x,center_y,holes)
                        if ends[bee_id] != None and get_area_of_detection(detection) >= average_hole_area:
                            ends[bee_id] = "too large"
                            #print("Detected an erroneous end point: Bee detection is too large.")
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
        
       
def get_average_hole_height(holes,image_size=(3840,2160)):
    (width,height) = image_size
    width_height_ratio = width/height

    hole_height_sum = 0
    for hole in holes:
        [top,left,bottom,right] = hole["bounding_box"]
        hole_height_sum += (bottom-top)/width_height_ratio
    return hole_height_sum/len(holes)
        
    
        
def enumerate_bee_detections(detection_map,working_dir,image_size=(3840,2160)):
    
    #TODO: Make sure that bees are kept being tracked even if for one or two frames the object detection algorithm didn't detect them
    
    (width,height) = image_size
    width_height_ratio = width/height
    bee_id_count = {}
    
    detected_holes_path= os.path.join(working_dir,"detected_holes.pkl")
    with open(detected_holes_path, 'rb') as f:
        holes = pickle.load(f)
    
    MAX_SQUARED_DISTANCE = pow(get_average_hole_height(holes,image_size)*1.4,2)
    print(MAX_SQUARED_DISTANCE)
    
        
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
                    distances.append((squared_distance,prev_index,curr_index))
            
            distances = sorted(distances, key = lambda tup: tup[0])
            
            already_assigned_prev_idxs = []
            for distance,prev_index,curr_index in distances:
                curr_detection = curr_detections[curr_index]
                prev_detection = prev_detections[prev_index]
                if distance < MAX_SQUARED_DISTANCE and not prev_index in already_assigned_prev_idxs and not "id" in curr_detection:
                    if curr_detection["name"] == "bee flying" and prev_detection["name"] == "bee flying":
                        curr_detection["id"] = -1
                    else:
                        if prev_detection["id"] != -1:
                            curr_detection["id"] = prev_detection["id"]
                        else:
                            curr_detection["id"] = current_bee_index
                            prev_detection["id"] = current_bee_index
                            current_bee_index += 1
                        if distance > 0.0015:
                            print(distance)
                            print(curr_detection["id"])
                    already_assigned_prev_idxs.append(prev_index)
                    
            
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
        if frame_number in detection_map and detection_map[frame_number] != "Skipped":
            for detection in detection_map[frame_number]:
                if bee_id_count[detection["id"]] < min_consecutive_frames_to_be_counted:
                    detection["id"] = -1
        frame_number += 1



        
        

    