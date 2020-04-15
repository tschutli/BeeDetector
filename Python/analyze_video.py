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

current_milli_time = lambda: int(round(time.time() * 1000))


MAX_SQUARED_DISTANCE = 0.01

image_size = constants.tensorflow_tile_size
#image_size = (1000,750)
min_confidence_score = 0.5

def analyze_video(trained_bee_model, trained_hole_model, input_video, working_dir):
    
    detect_bees(trained_bee_model,input_video,working_dir)
    
    
    
    
def detect_bees(trained_bee_model,input_video,working_dir):

    detection_map = {}
    
    last_24_frames = [None] * 24
    
    skip_counter = -1
    
    detection_graph = get_detection_graph(trained_bee_model)
    with detection_graph.as_default():
        with tf.Session() as sess:
            tensor_dict = get_tensor_dict(image_size)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0') 
            
            
            cap = cv2.VideoCapture(input_video)
            
            no_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            #frame_number = 0
            
            
            
            for frame_number in progressbar.progressbar(range(0,no_of_frames)):
            #while frame_number < no_of_frames:
                #start = current_milli_time()
                    
                ret, original_image = cap.read()
                if not ret:
                    break
                
                last_24_frames[frame_number%24] = original_image

                image = cv2.resize(original_image, image_size)
                
                #Skip frame if previously no bee was detected:
                if skip_counter>=0 and skip_counter < 12:
                    skip_counter += 1
                    detection_map[frame_number] = "Skipped"
                    continue                    
                
                
                
                 
                detections = get_detections(sess,image,image_tensor,tensor_dict)
                detection_map[frame_number] = detections
                
                crop_out_detections(original_image,frame_number,detections,working_dir)
                
                
                if not detections:
                    skip_counter = 0
                elif skip_counter == 12:
                    skip_counter = -1
                    
                    for i in range(1,13):
                        original_image = last_24_frames[(frame_number-i)%24]
                        image = cv2.resize(original_image, image_size)
                        detections = get_detections(sess, image, image_tensor, tensor_dict)
                        detection_map[frame_number-i] = detections
                        crop_out_detections(original_image,frame_number-i,detections,working_dir)
                        if not detections:
                            break
                               
                else:
                    skip_counter = -1
                
            enumerate_detections(detection_map)
            visualize(input_video,detection_map,"C:/Users/johan/Desktop/test.MP4")

            


            '''
            #Prediction part
            image_np = load_image_into_numpy_array(image)

            output_dict = sess.run(tensor_dict,feed_dict={image_tensor: np.expand_dims(image_np, 0)})
        
        
            detections = []
            count = 0
            for i,score in enumerate(output_dict['detection_scores']):
                if score >= min_confidence_score:
                    count += 1
                    top = round(output_dict['detection_boxes'][i][0] * original_height)
                    left = round(output_dict['detection_boxes'][i][1] * original_width)
                    bottom = round(output_dict['detection_boxes'][i][2] * original_height)
                    right = round(output_dict['detection_boxes'][i][3] * original_width)
                    detections.append({"bounding_box": [top,left,bottom,right], "score": float(score)})
            '''




def enumerate_detections(detection_map):
    
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
                    
        
        frame_number += 1
    

def crop_out_detections(image, frame_number, detections, output_dir):
    
    #saving cropped out detections:
    for index,detection in enumerate(detections):
        
        bee_tile_path = os.path.join(working_dir,"frame" + str(frame_number) + "_detection" + str(index) + ".png")
        
        height, width = image.shape[:2]

        [top,left,bottom,right] = detection["bounding_box"]
        top = int(top*height)
        bottom = int(bottom*height)
        left = int(left*width)
        right = int(right*width)
        
        bee_tile = image[top:bottom,left:right]
        
        cv2.imwrite(bee_tile_path, bee_tile)



def visualize(input_video,detection_map,output_path):
    
    cap = cv2.VideoCapture(input_video)
            
    out = cv2.VideoWriter(output_path,cv2.VideoWriter_fourcc(*'MP4V'), 25, image_size)

    no_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    for frame_number in progressbar.progressbar(range(0,no_of_frames)):
            
        ret, image = cap.read()
        if not ret:
            break
        
        image = cv2.resize(image, image_size)
        
        if detection_map[frame_number] and detection_map[frame_number] != "Skipped":
            for detection in detection_map[frame_number]:
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
                
                cv2.putText(image, str(detection["id"]), (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, rectangle_color, 2)
                
                    
        out.write(image)
        
        
    out.release()
    
        
        

def get_detections(sess,image, image_tensor, tensor_dict):
    #print("3: " + str(current_milli_time()-start))
    #frame_number += 1
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #print("4: " + str(current_milli_time()-start))
    image = Image.fromarray(image)
    
    #print("5: " + str(current_milli_time()-start))


    #(im_width, im_height) = image.size
    
    image_np = np.asarray(image) 
    #print("6: " + str(current_milli_time()-start))

    output_dict = sess.run(tensor_dict,feed_dict={image_tensor: np.expand_dims(image_np, 0)})
    
    #print("7: " + str(current_milli_time()-start))

    output_dict = clean_output_dict(output_dict)


    #print("8: " + str(current_milli_time()-start))

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
    return detections
    #print("9: " + str(current_milli_time()-start))
    #print()


def load_image_into_numpy_array(image):
    """
    Helper function that loads an image into a numpy array.
  
    Parameters:
        image (PIL image): a PIL image
      
    Returns:
        np.array: a numpy array representing the image
    """
    (im_width, im_height) = image.size    
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

    

def get_detection_graph(PATH_TO_FROZEN_GRAPH):
    """
    Reads the frozen detection graph into memory.
    
    Parameters:
        PATH_TO_FROZEN_GRAPH (str): path to the directory containing the frozen
            graph files.
    
    Returns:
        A tensorflow graph instance with which the prediction algorithm can be run.
    """
    detection_graph = tf.Graph()
    with detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
    return detection_graph


def get_tensor_dict(tile_size):
  """
  Helper function that returns a tensor_dict dictionary that is needed for the 
  prediction algorithm.
  
  Parameters:
      tile_size (int): The size of the tiles on which the prediction algorithm is
          run on.

  Returns:
      dict: The tensor dictionary
  
  """
      # Get handles to input and output tensors
  ops = tf.get_default_graph().get_operations()
  all_tensor_names = {output.name for op in ops for output in op.outputs}
  tensor_dict = {}
  for key in ['num_detections', 'detection_boxes', 'detection_scores','detection_classes', 'detection_masks']:
    tensor_name = key + ':0'
    if tensor_name in all_tensor_names:
      tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
  if 'detection_masks' in tensor_dict:
    # The following processing is only for single image
    detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
    detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
    # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
    real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
    detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
    detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
        detection_masks, detection_boxes, tile_size[1], tile_size[0])
    detection_masks_reframed = tf.cast(
        tf.greater(detection_masks_reframed, 0.5), tf.uint8)
    # Follow the convention by adding back the batch dimension
    tensor_dict['detection_masks'] = tf.expand_dims(
        detection_masks_reframed, 0)
    
  return tensor_dict

def clean_output_dict(output_dict):
    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]
    if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict


if __name__== "__main__":
    
    bee_model_path = "C:/Users/johan/Desktop/Agroscope_working_dir/trained_inference_graphs/output_inference_graph_v1.pb/frozen_inference_graph.pb"
    input_video = "C:/Users/johan/Desktop/MVI_0003_Trim2.MP4"
    working_dir = "C:/Users/johan/Desktop/test"
    analyze_video(bee_model_path, "", input_video,working_dir)
    
    