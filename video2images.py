# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 12:52:07 2020

@author: johan
"""

import cv2
import os
import multiprocessing as mp
from itertools import repeat
import signal


'''
def convert(video_path, output_folder,out_size=(640,480)):
    vidcap = cv2.VideoCapture(video_path)
    num_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    for count in progressbar.progressbar(range(0,num_frames)):
        success,image = vidcap.read()
        out_path = os.path.join(output_folder,"frame%d.png" % count)
        image = cv2.resize(image, (640,480))
        cv2.imwrite(out_path, image)
    
    
def get_frames_every_ms(video_path, output_folder,every_ms=1000/25.0, out_size=(640,480)):
    vidcap = cv2.VideoCapture(video_path)
    num_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    duration = num_frames/fps
    frames_to_read = int(duration/(every_ms/1000.0))

    

    for count in progressbar.progressbar(range(0,frames_to_read)):
        vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*every_ms))    # added this line 
        success,image = vidcap.read()
        out_path = os.path.join(output_folder,"frame%d.png" % count)
        image = cv2.resize(image, (640,480))
        cv2.imwrite(out_path, image)
        




def make_annotation_frames(video_path, output_folder):
    vidcap = cv2.VideoCapture(video_path)
    num_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    
    for count in progressbar.progressbar(range(0,num_frames)):
        success,image = vidcap.read()
        out_path = os.path.join(output_folder,"sec_" + str(int(count/fps)) + "_frame_" + str(count % fps) + ".png")
        #image = cv2.resize(image, (640,480))
        cv2.imwrite(out_path, image)
'''


def process_video(group_number, num_processes, video_path, output_folder, out_size=(640,480), skip_frames=0):
    
    cap = cv2.VideoCapture(video_path)
    no_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
    start_frame = int(no_of_frames / num_processes)* group_number
    if start_frame % (skip_frames+1) != 0:
        start_frame += (skip_frames+1 - start_frame % (skip_frames+1))
    end_frame = int(no_of_frames / num_processes)* (group_number+1)
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_number = start_frame
    
    
    try:
        while frame_number < end_frame:
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, image = cap.read()
            if not ret:
                break
            
            minutes = str(int(frame_number/fps/60)).zfill(3)
            seconds = str(int(frame_number/fps) % 60).zfill(2)
            frame = str(frame_number % fps).zfill(2)
            
            
            out_path = os.path.join(output_folder,"min_" + minutes+ "_sec_" + seconds + "_frame_" + frame + ".jpg")
            image = cv2.resize(image, out_size)
            cv2.imwrite(out_path, image)
            if (frame_number-start_frame) % 100 == 0:
                print("Thread " + str(group_number) + " progress: " + str(frame_number-start_frame) + " / " + str(end_frame-start_frame))
            frame_number += 1 + skip_frames


    except:
        print("Thread " + str(group_number) + ": ERROR!!")
        # Release resources
        cap.release()

    # Release resources
    cap.release()
    print("Thread " + str(group_number) + ": DONE!")


    


def extract_frames_parallel(video_path,output_folder,out_size=(3840,2160),skip_frames=0, num_processes=mp.cpu_count()):
    
    original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    pool = mp.Pool(num_processes)
    signal.signal(signal.SIGINT, original_sigint_handler)

    try:
        process_arguments = zip(range(num_processes),repeat(num_processes),repeat(video_path),repeat(output_folder),repeat(out_size),repeat(skip_frames))
        pool.starmap(process_video, process_arguments)
    except KeyboardInterrupt:
        print("Caught KeyboardInterrupt, terminating workers")
        pool.terminate()
    else:
        print("Normal termination")
        pool.close()

        

    
    


if __name__ == '__main__':
    extract_frames_parallel("C:/Users/johan/Desktop/MVI_0003.MP4","D:/Frames",(3840,2160),2)
    #make_annotation_frames("D:/MVI_0003.MP4","C:/Users/johan/Desktop/test")
    #get_frames_every_ms("D:/MVI_0003.MP4","C:/Users/johan/Desktop/test",every_ms=1000,out_size=(640,480))
    