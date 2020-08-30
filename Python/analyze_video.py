# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 12:52:15 2020

@author: johan
"""

import cv2
import progressbar
import os
import pickle
import queue
import threading
import frame_reader
import predictor
from threading import Event
from concurrent.futures import ThreadPoolExecutor
import hole_analysis
import colors_detector
import numbers_predictor
import extract_stats
import yolo_predictor


tensorflow_image_size = (1024,576)



class ProgressHelper(object): 
    input_videos = None
    progress_callback = None
    progress_bar_key = None
    progress_map = {}
      
    def __init__(self, input_videos, progress_callback, progress_bar_key): 
        self.input_videos = input_videos
        self.progress_callback = progress_callback
        self.progress_bar_key = progress_bar_key
        self.progress_map = {}
        for input_video in input_videos:
            self.progress_map[input_video] = 0.0

      
    def get_overall_progress(self): 
        progress_sum = 0.0
        for input_video in self.progress_map.keys():
            progress_sum += self.progress_map[input_video]
        return progress_sum / len(self.progress_map.keys())

    
    def control_progress_callback(self,progress,input_video):
        if self.progress_callback == None:
            return
        if type(progress) == float:
            self.progress_map[input_video] = progress
            self.progress_callback((self.progress_bar_key,self.get_overall_progress()))
        else:
            self.progress_callback(progress + ": " + input_video)






def analyze_videos(input_videos, working_dir, visualize=True, progress_callback=None, pause_event=None,config_file_path=None):
    
    #Load config file parameters
    if not config_file_path:
        import config
    else:
        import importlib.util
        spec = importlib.util.spec_from_file_location("config", config_file_path)
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)
    
        
    working_dirs = []
    for input_video in input_videos:
        video_dir = os.path.join(working_dir, os.path.basename(input_video))
        while video_dir in working_dirs:
            video_dir += "2"
        working_dirs.append(video_dir)
        os.makedirs(video_dir,exist_ok=True)
          
    
    detect_bees(config.bee_model_path,input_videos,working_dirs, progress_callback, pause_event, config.num_worker_threads)
    
    detect_holes(config.hole_model_path,input_videos,working_dirs,progress_callback, pause_event, config.num_worker_threads)

    detect_colors(config.color_model_path,working_dirs,progress_callback,pause_event, config.num_worker_threads)
    
    detect_numbers(config.number_model_path,working_dirs,progress_callback,pause_event)
    
    if pause_event != None and pause_event.is_set():
        return

    extract_stats.extract_stats(working_dirs,config)
        
    if visualize:
        visualize_videos(input_videos,working_dirs,progress_callback, pause_event,config.num_worker_threads,config.visualization_video_size)
    
    if pause_event != None and not pause_event.is_set():
        progress_callback("Success. All videos are analyzed!")


def detect_numbers(trained_numbers_model,working_dirs,progress_callback=None,pause_event=None):
    progress_helper = ProgressHelper(working_dirs,progress_callback,"detect_numbers")
    numbers_predictor.start(trained_numbers_model,working_dirs,pause_event,progress_helper.control_progress_callback)


def detect_colors(trained_colors_model,working_dirs,progress_callback=None,pause_event=None,num_threads=2):
    progress_helper = ProgressHelper(working_dirs,progress_callback,"detect_colors")
    
    
    frame_queue = queue.PriorityQueue()
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for working_dir in working_dirs:
            labels = ["green","white","yellow"]
            future = executor.submit(colors_detector.detect_colors,working_dir,frame_queue,labels,progress_helper.control_progress_callback, pause_event)
            
            futures.append(future)
        
        
        predictor_stop_event = Event()
        predictor_thread = threading.Thread(target=yolo_predictor.start,args=(trained_colors_model,frame_queue,predictor_stop_event,))
        predictor_thread.daemon=True
        predictor_thread.start()
        
        for future in futures:
            future.result()
        predictor_stop_event.set()
        predictor_thread.join()


    
def detect_holes(trained_hole_model, input_videos, working_dirs,progress_callback=None, pause_event=None,num_threads=2):
    
    
    progress_helper = ProgressHelper(working_dirs,progress_callback,"detect_holes")

    frame_queue = queue.PriorityQueue()
        
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for (input_video,working_dir) in zip(input_videos,working_dirs):
            future = executor.submit(hole_analysis.hole_frame_reader,working_dir,frame_queue,tensorflow_image_size,progress_helper.control_progress_callback, pause_event)
            
            futures.append(future)
        
        
        predictor_stop_event = Event()
        predictor_thread = threading.Thread(target=predictor.start,args=(trained_hole_model,frame_queue,predictor_stop_event,))
        predictor_thread.daemon=True
        predictor_thread.start()
        
        for future in futures:
            future.result()
        predictor_stop_event.set()
        predictor_thread.join()


    
    

        
        
    
    
    
    
def detect_bees(trained_bee_model,input_videos,working_dirs, progress_callback=None, pause_event=None,num_threads=2):
    
        
    
    progress_helper = ProgressHelper(input_videos,progress_callback,"detect_bees")

    
    frame_queue = queue.PriorityQueue()
        
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for (input_video,working_dir) in zip(input_videos,working_dirs):
            future = executor.submit(frame_reader.start, input_video, frame_queue, working_dir,tensorflow_image_size,progress_helper.control_progress_callback,pause_event)
            futures.append(future)
        
        
        predictor_stop_event = Event()
        predictor_thread = threading.Thread(target=predictor.start,args=(trained_bee_model,frame_queue,predictor_stop_event,))
        predictor_thread.daemon=True
        predictor_thread.start()
        
        for future in futures:
            future.result()
        predictor_stop_event.set()
        predictor_thread.join()
        
        



    
def visualize_videos(input_videos,working_dirs,progress_callback=None, pause_event=None,num_threads=2,image_size=tensorflow_image_size):
    
    progress_helper = ProgressHelper(input_videos,progress_callback,"visualize")
    
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        
        futures = []
        
        for (input_video,working_dir) in zip(input_videos,working_dirs):
            detection_map = {}
            with open(os.path.join(working_dir,"combined_detections.pkl"), 'rb') as f:
                detection_map = pickle.load(f)
            
            future = executor.submit(visualize, input_video, detection_map, os.path.join(working_dir,"visualization.MP4"),progress_helper.control_progress_callback,pause_event,image_size)
            futures.append(future)

        for future in futures:
            future.result()

            
    


def visualize(input_video,detection_map,output_path,progress_callback=None, pause_event=None,image_size=tensorflow_image_size):

    if pause_event != None and pause_event.is_set():
        return
    
    
    progress_callback("Starting Visualization of video", input_video)

    out = cv2.VideoWriter(output_path,cv2.VideoWriter_fourcc(*'MP4V'), 25, image_size)
    

    cap = cv2.VideoCapture(input_video)

    no_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_number = 0
    for frame_number in progressbar.progressbar(range(0,no_of_frames)):
            
        if frame_number % 100 == 0:
            progress_callback(frame_number/no_of_frames,input_video)
        
            if pause_event != None and pause_event.is_set():
                cap.release() 
                out.release()
                return
            
        ret, image = cap.read()
        if not ret:
            print("BREAK")
            break

        image = cv2.resize(image, image_size)
        
        if not frame_number in detection_map:
            print("Did not find frame " + str(frame_number) + " in detection_map.")
            continue
        
        if detection_map[frame_number] and detection_map[frame_number] != "Skipped":
            for detection in detection_map[frame_number]:
                [top,left,bottom,right] = detection["bounding_box"]
                top = int(top*image_size[1])
                bottom = int(bottom*image_size[1])
                left = int(left*image_size[0])
                right = int(right*image_size[0])
                rectangle_color = (0,0,255)
                if(detection["name"] == "bee"):
                    rectangle_color = (0,0,255)
                elif(detection["name"] == "bee flying"):
                    rectangle_color = (255,0,0)      
                
                image = cv2.rectangle(image, (left,top), (right,bottom), rectangle_color, 2)
                
                if detection["id"] != -1:
                    starting_point = detection["start"]
                    if starting_point == None or starting_point == "too large":
                        starting_point = "?"
                    end_point = detection["end"]
                    if end_point == None or end_point == "too large":
                        end_point = "?"
                    
                    if starting_point != "?" or end_point != "?":
                        bee_description = starting_point + " -> " + end_point
    
                        '''
                        if "color" in detection and detection["color"] != None:
                            bee_description = str(detection["color"]) + '{0:.2f}'.format(detection["color_score"]) +" " + bee_description
                        
                        if "number" in detection:
                            bee_description = str(detection["number"]) + '{0:.2f}'.format(detection["number_score"]) + " " + bee_description
                        '''
                        if "final_color" in detection and "final_number" in detection:
                            bee_description = str(detection["final_color"]) + str(detection["final_number"]) + ": " + bee_description
                        cv2.putText(image, bee_description, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 1, rectangle_color, 2)
                        
                        
                        

        #if detection_map[frame_number] != "Skipped":
        out.write(image)

    cap.release() 
    out.release()
    progress_callback(1.0,input_video)
    progress_callback("Finished visualizing video", input_video)

    

    


if __name__== "__main__":

    def progress_callback(progress):
        print(progress)
    
    pause_event = Event()
    from utils import constants
    input_videos = ["E:/input/MVI_0003.MP4"]
    working_dir = "E:/output"
    analyze_videos(input_videos, working_dir,progress_callback=progress_callback,pause_event=pause_event)
    
    