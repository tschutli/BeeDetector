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


current_milli_time = lambda: int(round(time.time() * 1000))


MAX_SQUARED_DISTANCE = 0.01

image_size = constants.tensorflow_tile_size
#image_size = (1000,750)
min_confidence_score = 0.5
min_consecutive_frames_to_be_counted = 3
num_threads = 7


def analyze_video(trained_bee_model, trained_hole_model, input_video, working_dir):
    
    detect_bees(trained_bee_model,input_video,working_dir)
    
    
    
    
    
def detect_bees(trained_bee_model,input_video,working_dir):
    '''
    start = current_milli_time()
    
    frame_queue = queue.PriorityQueue()
    
    worker_threads = []
    results = [{}] * num_threads

    for thread_id in range(num_threads):
        
        enqueue_thread = threading.Thread(target=frame_reader.start,args=(thread_id, num_threads, input_video, frame_queue, working_dir,results[thread_id],image_size,))
        enqueue_thread.daemon=True
        worker_threads.append(enqueue_thread)
        enqueue_thread.start()

    
    predictor_stop_event = Event()
    predictor_thread = threading.Thread(target=predictor.start,args=(trained_bee_model,frame_queue,image_size,predictor_stop_event,))
    predictor_thread.daemon=True
    predictor_thread.start()

    detection_map = {}
    
    for worker_thread in worker_threads:
        worker_thread.join()
    predictor_stop_event.set()
    
    for result in results:
        detection_map.update(result)


        
    
    print("Total Prediction Time: " + str(current_milli_time()-start))
    
    with open(os.path.join(working_dir,"detection_map.pkl"), 'wb') as f:
        pickle.dump(detection_map,f)
    '''
    
    detection_map = {}
    with open(os.path.join(working_dir,"detection_map.pkl"), 'rb') as f:
    
        detection_map = pickle.load(f)


    print(len(detection_map.keys()))
    
    count = 0
    fps = 25
    
    '''
    for frame_number in range(0,19404):
        if frame_number in detection_map.keys() and detection_map[frame_number] == "Skipped":
           count += 1
        
        if detection_map[frame_number] and detection_map[frame_number] != "Skipped":
            
            minutes = str(int(frame_number/fps/60)).zfill(3)
            seconds = str(int(frame_number/fps) % 60).zfill(2)
            frame = str(frame_number % fps).zfill(2)
            print(minutes + ":" + seconds + "." + frame)
    '''
    print(count)
    

    enumerate_detections(detection_map)
    visualize(input_video,detection_map,os.path.join(working_dir,"test.MP4"))
    
            





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
                    (x, y) = ((right+left)/2,(bottom-top)/2)
                    
                    distances = []
                    
                    for prev_index,prev_detection in enumerate(prev_detections):
                        [top,left,bottom,right] = prev_detection["bounding_box"]
                        (prev_x, prev_y) = ((right+left)/2,(bottom-top)/2)
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
    


def visualize(input_video,detection_map,output_path):
    #start = current_milli_time()
    #cap = cv2.VideoCapture(input_video)
    #cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    out = cv2.VideoWriter(output_path,cv2.VideoWriter_fourcc(*'MP4V'), 25, image_size)

    fps = 25
    

    for thread_id in range(num_threads):
        
        cap = cv2.VideoCapture(input_video)

        no_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        start_frame = int(no_of_frames / num_threads)* thread_id
        end_frame = int(no_of_frames / num_threads)* (thread_id+1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        for frame_number in progressbar.progressbar(range(start_frame,end_frame)):
    
        
        #for frame_number in progressbar.progressbar(range(0,no_of_frames)):
                
            start = current_milli_time()
            cap_frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)
            if cap_frame_number != frame_number:
                print("ERROR")
                print(cap_frame_number)
                print(frame_number)
                time.sleep(1)
    
            ret, image = cap.read()
            if not ret:
                print("BREAK")
                break
            #print("",flush=True)
            
            #print("1: " + str(current_milli_time()-start))
    
            image = cv2.resize(image, image_size)
            #print("2: " + str(current_milli_time()-start), flush=True)
            
            
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
                        minutes = str(int(frame_number/fps/60)).zfill(3)
                        seconds = str(int(frame_number/fps) % 60).zfill(2)
                        frame = str(frame_number % fps).zfill(2)
                        print(minutes + ":" + seconds)
                        
    
            #print("3: " + str(current_milli_time()-start), flush=True)
            #if detection_map[frame_number] != "Skipped":
            out.write(image)
            #print("4: " + str(current_milli_time()-start), flush=True)
            #print("", flush=True)

        
    out.release()
    

    



if __name__== "__main__":
    
    bee_model_path = "G:/Johannes/Test/Working_dir_3/trained_inference_graphs/output_inference_graph_v1.pb/frozen_inference_graph.pb"
    input_video = "G:/Johannes/Test/MVI_0003.MP4"
    working_dir = "G:/Johannes/Test/test"
    analyze_video(bee_model_path, "", input_video,working_dir)
    
    