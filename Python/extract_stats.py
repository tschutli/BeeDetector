# -*- coding: utf-8 -*-
"""
Created on Fri May  8 14:20:00 2020

@author: johan
"""

import os
import pickle
from datetime import timedelta


min_consecutive_frames_to_be_counted = 3

#A start of a flight is only counted if the area of the bee detection bounding 
#box is smaller than hole_area_factor*average_hole_area
hole_area_factor = 1.2

#A bee will only be tracked over time if it is not further apart than 
#(MAX_TRACKING_DISTANCE_FACTOR*average_hole_height) in two consecutive frames
MAX_TRACKING_DISTANCE_FACTOR = 1.4

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

        write_starts_and_ends_to_file(detection_map,working_dir)
        
        

def write_starts_and_ends_to_file(detection_map,working_dir):
    
    def is_id_in_frame(bee_id, frame_number):
        if frame_number in detection_map and detection_map[frame_number] != "Skipped":
            for detection in detection_map[frame_number]:
                if detection["id"] == bee_id:
                    return True
        return False

    
    csv_file = os.path.join(working_dir,"all_events.csv")
    if os.path.exists(csv_file):
        os.remove(csv_file)
    
    
    with open(csv_file, 'a') as f:
        f.write("TIME,BEE,ACTION,HOLE\n")
        fps = 25
        
        frame_number = 0
        while frame_number in detection_map:
            if detection_map[frame_number] and detection_map[frame_number] != "Skipped":
                for detection in detection_map[frame_number]:
                    bee_id = detection["id"]
                    if bee_id == -1:
                        continue
                    action = None
                    hole = None
                    if not is_id_in_frame(bee_id,frame_number-1) and detection["start"] != None and detection["start"] != "too large":
                        action = "Leave"
                        hole = detection["start"]
                    elif not is_id_in_frame(bee_id,frame_number+1) and detection["end"] != None and detection["end"] != "too large":
                        action = "Enter"
                        hole = detection["end"]
                    
                    if action != None:
                        seconds = frame_number*(1/fps)
                        time = timedelta(seconds=seconds)
                        name = "?"
                        if "final_color" in detection and "final_number" in detection:
                            name = detection["final_color"] + str(detection["final_number"])
                        f.write(str(time) + "," + name + "," + action + "," + hole+"\n")
            frame_number+=1
                        
                        
                        


        
def filter_numbers_and_colors(detection_map):
    
    id2color = {}
    id2number = {}
    frame_number = 0
    while frame_number in detection_map:
        if detection_map[frame_number] and detection_map[frame_number] != "Skipped":
            for detection in detection_map[frame_number]:
                if "color" in detection and detection["color"] != None:
                    bee_id = detection["id"]
                    if bee_id == -1:
                        continue
                    if detection["color_score"] > 0.9:
                        if not bee_id in id2color:
                            id2color[bee_id] = []
                            id2number[bee_id] = []
                        id2color[bee_id].append((detection["color"],detection["color_score"]))
                        id2number[bee_id].append((detection["number"],detection["number_score"]))
                        
        frame_number += 1
    
    def find_best_prediction(tuple_list):
        
        total_scores = {}
        for (label,score) in tuple_list:
            if score < 0.9 or label == 0 or label == "0":
                continue
            if not label in total_scores:
                total_scores[label] = score
            else:
                total_scores[label] += score
        if not total_scores:
            return None
        return max(total_scores, key=total_scores.get)
            
        '''
        sorted_by_score = sorted(tuple_list, key=lambda tup: tup[1],reverse=True)
        return sorted_by_score[0]
        '''        
        
        
    for bee_id in id2color.keys():
        id2color[bee_id] = find_best_prediction(id2color[bee_id])
    
    for bee_id in id2number.keys():
        id2number[bee_id] = find_best_prediction(id2number[bee_id])

        
    frame_number = 0
    while frame_number in detection_map:
        if detection_map[frame_number] and detection_map[frame_number] != "Skipped":
            for detection in detection_map[frame_number]:
                bee_id = detection["id"]
                if bee_id in id2color and id2color[bee_id] != None:
                    detection["final_color"] = id2color[bee_id]
                if bee_id in id2number and id2number[bee_id] != None:
                    detection["final_number"] = id2number[bee_id]
                             
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
                        if starts[bee_id] != None and get_area_of_detection(detection) >= average_hole_area*hole_area_factor:
                            starts[bee_id] = "too large"
                    else:
                        starts[bee_id] = None
                if not is_id_in_frame(bee_id,frame_number+1):
                    if detection["name"] == "bee":
                        #Bee is sitting
                        ends[bee_id] = get_hole_id_at_position(center_x,center_y,holes)
                        if ends[bee_id] != None and get_area_of_detection(detection) >= average_hole_area*hole_area_factor:
                            ends[bee_id] = "too large"
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
                '''
                if starts[bee_id] == ends[bee_id]:
                    detection["id"] = -1
                else:
                    detection["start"] = starts[bee_id]
                    detection["end"] = ends[bee_id]   
                '''
                
                
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
    
    MAX_SQUARED_DISTANCE = pow(get_average_hole_height(holes,image_size)*MAX_TRACKING_DISTANCE_FACTOR,2)
    
        
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
            
            distances_current_frame = []
            for index1,detection1 in enumerate(curr_detections):
                for index2,detection2 in enumerate(curr_detections):
                    if index1 == index2:
                        continue
                    [top,left,bottom,right] = detection1["bounding_box"]
                    (x1, y1) = ((right+left)/2,(bottom+top)/2)
                    [top,left,bottom,right] = detection2["bounding_box"]
                    (x2, y2) = ((right+left)/2,(bottom+top)/2)
                    squared_distance = pow(x1-x2,2) + pow((y1-y2)/width_height_ratio,2)
                    distances_current_frame.append((squared_distance,index1,index2))
            
            distances_current_frame = sorted(distances_current_frame, key = lambda tup: tup[0])
            distances = sorted(distances, key = lambda tup: tup[0])
            
            
            
            def is_another_detection_close(index,max_distance):
                for (d,index1,index2) in distances_current_frame:
                    if index1 == index and d<=max_distance:
                        return True
                return False
            
            
            
            already_assigned_prev_idxs = []
            for distance,prev_index,curr_index in distances:
                curr_detection = curr_detections[curr_index]
                prev_detection = prev_detections[prev_index]
                if distance < MAX_SQUARED_DISTANCE and not prev_index in already_assigned_prev_idxs and not "id" in curr_detection:
                    #The two bees are close and both have not been assigned yet
                    
                    if curr_detection["name"] == "bee flying" and prev_detection["name"] == "bee flying":
                        
                        if is_another_detection_close(curr_index,MAX_SQUARED_DISTANCE*4):
                            curr_detection["id"] = -1
                        elif prev_detection["id"] != -1:
                            curr_detection["id"] = prev_detection["id"]
                        else:
                            curr_detection["id"] = current_bee_index
                            current_bee_index += 1
                            
                    
                    else:
                        if prev_detection["id"] != -1:
                            curr_detection["id"] = prev_detection["id"]
                        else:
                            curr_detection["id"] = current_bee_index
                            prev_detection["id"] = current_bee_index
                            current_bee_index += 1
                    already_assigned_prev_idxs.append(prev_index)
                    
            
            
            for curr_index,curr_detection in enumerate(curr_detections):
                if not "id" in curr_detection:
                    if is_another_detection_close(curr_index,MAX_SQUARED_DISTANCE*4):
                        curr_detection["id"] = -1
                    else:
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



        
        

    