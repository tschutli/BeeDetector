# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 12:52:15 2020

@author: johan
"""

import tensorflow as tf
import numpy as np
from object_detection.utils import ops as utils_ops
from utils import constants
import cv2
import progressbar
import time
from PIL import Image
import os
from utils import eval_utils
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


def analyze_video(trained_bee_model, trained_hole_model, input_video, working_dir):
    
    detect_bees(trained_bee_model,input_video,working_dir)
    
    
    
def thread_fill_queue(queue,input_video):
    cap = cv2.VideoCapture(input_video)
    ret, original_image = cap.read()
    while ret:
        queue.put(original_image)
        ret, original_image = cap.read()
        print("qsize: " + str(queue.qsize()))


    
    
    
def detect_bees(trained_bee_model,input_video,working_dir):

    num_threads = 3
    frame_queue = queue.Queue(100)
    
    worker_threads = []

    for thread_id in range(num_threads):
        
        enqueue_thread = threading.Thread(target=frame_reader.start,args=(thread_id, num_threads, input_video, frame_queue, working_dir,image_size,))
        enqueue_thread.daemon=True
        worker_threads.append(enqueue_thread)
        enqueue_thread.start()

    
    predictor_stop_event = Event()
    predictor_thread = threading.Thread(target=predictor.start,args=(trained_bee_model,frame_queue,image_size,predictor_stop_event,))
    predictor_thread.daemon=True
    predictor_thread.start()

    for worker_thread in worker_threads:
        woker_result = worker_thread.join()
        #TODO merge worker results
        
    predictor_stop_event.set()
    
    print("AAALL DONE")
    
         
        
    return

    with open(os.path.join(working_dir,"detection_map.pkl"), 'rb') as f:
    
        detection_map = pickle.load(f)

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
    start = current_milli_time()
    cap = cv2.VideoCapture(input_video)
            

    out = cv2.VideoWriter(output_path,cv2.VideoWriter_fourcc(*'MP4V'), 25, image_size)

    no_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    for frame_number in progressbar.progressbar(range(0,no_of_frames)):
            
        start = current_milli_time()


        ret, image = cap.read()
        if not ret:
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
        
        #print("3: " + str(current_milli_time()-start), flush=True)
        if detection_map[frame_number] != "Skipped":
            out.write(image)
        #print("4: " + str(current_milli_time()-start), flush=True)
        #print("", flush=True)

        
    out.release()
    
        
        

def get_detections(sess,image, image_tensor, tensor_dict):
    
    start = current_milli_time()

    print("3: " + str(current_milli_time()-start))
    #frame_number += 1
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print("4: " + str(current_milli_time()-start))
    image = Image.fromarray(image)
    
    print("5: " + str(current_milli_time()-start))


    #(im_width, im_height) = image.size
    
    image_np = np.asarray(image) 
    
    print("5.4: " + str(current_milli_time()-start))

    image_expand = np.expand_dims(image_np, 0)
    print("6: " + str(current_milli_time()-start))

    output_dict = sess.run(tensor_dict,feed_dict={image_tensor: image_expand})
    
    print("7: " + str(current_milli_time()-start))

    output_dict = clean_output_dict(output_dict)


    print("8: " + str(current_milli_time()-start))

    #print(image_np.dtype)

    
    detections = []
    count = 0
    for i,score in enumerate(output_dict['detection_scores']):
        if score >= min_confidence_score:
            count += 1
            top = output_dict['detection_boxes'][i][0]
            left = output_dict['detection_boxes'][i][1]
            bottom = output_dict['detection_boxes'][i][2]
            right = output_dict['detection_boxes'][i][3]
            detection_class = output_dict['detection_classes'][i]
            detections.append({"bounding_box": [top,left,bottom,right], "score": float(score), "class": detection_class})
    
    detections = eval_utils.non_max_suppression(detections,0.5)
    print("9: " + str(current_milli_time()-start))

    return detections
    #print("9: " + str(current_milli_time()-start))
    #print()


    



if __name__== "__main__":
    
    bee_model_path = "C:/Users/johan/Desktop/Agroscope_working_dir/trained_inference_graphs/output_inference_graph_v1.pb/frozen_inference_graph.pb"
    input_video = "C:/Users/johan/Desktop/MVI_0003.MP4"
    working_dir = "C:/Users/johan/Desktop/test"
    analyze_video(bee_model_path, "", input_video,working_dir)
    
    