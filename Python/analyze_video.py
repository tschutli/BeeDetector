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
import hole_analyze




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
            future = executor.submit(hole_analyze.hole_frame_reader,working_dir,frame_queue,image_size)
            
            futures.append(future)
        
        
        predictor_stop_event = Event()
        predictor_thread = threading.Thread(target=predictor.start,args=(trained_hole_model,frame_queue,image_size,predictor_stop_event,))
        predictor_thread.daemon=True
        predictor_thread.start()
        
        for future in futures:
            future.result()
        predictor_stop_event.set()


    
    

        
        
    
    
    
    
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
    
    