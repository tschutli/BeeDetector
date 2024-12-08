# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 13:00:29 2020

@author: johan
"""
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import time
import statistics
import queue
from dataclasses import dataclass, field
from typing import Any
from PIL import Image


@dataclass(order=True)
class PrioritizedItem:
    priority: int
    item: Any=field(compare=False)

current_milli_time = lambda: int(round(time.time() * 1000))


def start(trained_model,frame_queue,stop_event):
    
    stats = []
    stats_wait = []

    
    detection_graph = get_detection_graph(trained_model + "/frozen_inference_graph.pb")
    with detection_graph.as_default():
        with tf.Session() as sess:
            
            tensor_dict = get_tensor_dict()
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0') 

            #dummy run
            image = Image.new('RGB', (1024,576))
            image_np = np.asarray(image)         
            image_expand = np.expand_dims(image_np, 0)
            sess.run(tensor_dict,feed_dict={image_tensor: image_expand})
            
            while not stop_event.is_set():
                start = current_milli_time()
                try:
                    (image_expand,detections,is_done) = frame_queue.get(timeout = 1).item
                    
                    stats_wait.append(current_milli_time()-start)
    
                    #print("wait: " + str(current_milli_time()-start), flush=True)
                    start = current_milli_time()
    
                    out_dict = sess.run(tensor_dict,feed_dict={image_tensor: image_expand})
                    clean_pred_dict(out_dict,detections)
                    is_done.set()
                    stats.append(current_milli_time()-start)
                    #print("done: " + str(current_milli_time()-start), flush=True)
                    if len(stats)%100 == 0:
                        print("median_predict: " + str(statistics.median(stats)))
                        print("median_wait: " + str(statistics.median(stats_wait)))
                
                except queue.Empty:
                    continue
                    
                
            print("Quitting prediction Thread")










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


def get_tensor_dict():
  """
  Helper function that returns a tensor_dict dictionary that is needed for the 
  prediction algorithm.
  
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
  return tensor_dict



def clean_pred_dict(pred_dict, output_dict):
    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(pred_dict['num_detections'][0])
    output_dict['detection_classes'] = pred_dict['detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = pred_dict['detection_boxes'][0]
    output_dict['detection_scores'] = pred_dict['detection_scores'][0]
    if 'detection_masks' in pred_dict:
        output_dict['detection_masks'] = pred_dict['detection_masks'][0]
    return output_dict
