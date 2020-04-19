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


current_milli_time = lambda: int(round(time.time() * 1000))



image_size = constants.tensorflow_tile_size
#image_size = (1000,750)
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
    
    
    
    
    
    
def detect_bees(trained_bee_model,input_videos,working_dirs):
    
        
    #start = current_milli_time()
    
    frame_queue = queue.PriorityQueue()
        
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for (input_video,working_dir) in zip(input_videos,working_dirs):
            if os.isfile(os.path.join(working_dir,"detection_map.pkl")):
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
        

        
    return

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
    

    #enumerate_detections(detection_map)
    visualize(input_video,detection_map,os.path.join(working_dir,"test.MP4"))
    
            
    


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
    
    bee_model_path = "C:/Users/johan/Desktop/Agroscope_working_dir/trained_inference_graphs/output_inference_graph_v1.pb/frozen_inference_graph.pb"
    input_videos = ["C:/Users/johan/Desktop/test.MP4","C:/Users/johan/Desktop/MVI_0003_Trim.mp4"]
    working_dir = "C:/Users/johan/Desktop/test"
    analyze_videos(bee_model_path, "", input_videos, working_dir)
    
    