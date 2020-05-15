# -*- coding: utf-8 -*-
"""
Created on Mon May 11 15:55:35 2020

@author: johan
"""

import colorsys
import os
from timeit import default_timer as timer

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw
#from object_detection.utils import visualization_utils
from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
from keras.utils import multi_gpu_model
from utils import file_utils
import time
import progressbar
current_milli_time = lambda: int(round(time.time() * 1000))
import statistics
import queue
from dataclasses import dataclass, field
from typing import Any
from utils import constants

@dataclass(order=True)
class PrioritizedItem:
    priority: int
    item: Any=field(compare=False)

current_milli_time = lambda: int(round(time.time() * 1000))


def start(trained_model,frame_queue,stop_event):
    
    stats = []
    stats_wait = []

    yolo = YOLO(model_path=trained_model)



    while not stop_event.is_set():
        start = current_milli_time()
        try:
            (image_expand,image_size,detections,is_done) = frame_queue.get(timeout = 1).item
            
            stats_wait.append(current_milli_time()-start)


            start = current_milli_time()
            out_boxes,out_scores,out_classes = yolo.detect_fast(image_size,image_expand)
            detections["detection_boxes"] = out_boxes
            detections["detection_scores"] = out_scores
            detections["detection_classes"] = out_classes
            is_done.set()
            stats.append(current_milli_time()-start)
            
            if len(stats)%100 == 0:
                print("median_predict: " + str(statistics.median(stats)))
                print("median_wait: " + str(statistics.median(stats_wait)))
                print("avg_wait: " + str(statistics.mean(stats_wait)))

        
        except queue.Empty:
            continue
            
        
    print("Quitting prediction Thread")


'''
def predict(model_path,images_to_predict,output_folder,classes,visualize_groundtruths=False,visualize_predictions=True,visualize_scores=True,visualize_names=True):
    
    yolo = YOLO(model_path=model_path)
    
    all_images = file_utils.get_all_image_paths_in_folder(images_to_predict)
    
    for image_path in progressbar.progressbar(all_images):
        
        image = Image.open(image_path)
        width,height= image.size
        out_boxes,out_scores,out_classes = yolo.detect_image(image)
    
        detections = []
                
        for box,score,clazz in zip(out_boxes,out_scores,out_classes):
            top = round(box[0])
            left = round(box[1])
            bottom = round(box[2])
            right = round(box[3])
            detections.append({"bounding_box": [top,left,bottom,right], "score": float(score), "name": classes[clazz]})

        predictions_out_path = os.path.join(output_folder, os.path.basename(image_path)[:-4] + ".xml")
        file_utils.save_annotations_to_xml(detections, image_path, predictions_out_path)
        
        
        #copy the ground truth annotations to the output folder if there is any ground truth
        ground_truth = get_ground_truth_annotations(image_path)
        if ground_truth:
            #draw ground truth
            if visualize_groundtruths:
                for detection in ground_truth:
                    [top,left,bottom,right] = detection["bounding_box"]
                    col = "black"
                    visualization_utils.draw_bounding_box_on_image(image,top,left,bottom,right,display_str_list=(),thickness=1, color=col, use_normalized_coordinates=False)          

            ground_truth_out_path = os.path.join(output_folder, os.path.basename(image_path)[:-4] + "_ground_truth.xml")
            file_utils.save_annotations_to_xml(ground_truth,image_path,ground_truth_out_path)
        

        for detection in detections:
            if visualize_predictions:
                col = 'LightCyan'
                [top,left,bottom,right] = detection["bounding_box"]
                score_string = str('{0:.2f}'.format(detection["score"]))
                vis_string_list = []
                if visualize_scores:
                    vis_string_list.append(score_string)
                if visualize_names:
                    vis_string_list.append(detection["name"])                            
                visualization_utils.draw_bounding_box_on_image(image,top,left,bottom,right,display_str_list=vis_string_list,thickness=1, color=col, use_normalized_coordinates=False)          
        
        if visualize_groundtruths or visualize_predictions:
            image_output_path = os.path.join(output_folder, os.path.basename(image_path))
            image.save(image_output_path)
'''


class YOLO(object):
    _defaults = {
        "model_path": 'C:/Users/johan/Downloadds/ep037-loss17.009-val_loss15.844.h5',
        "score" : 0.3,
        "iou" : 0.45,
        "model_image_size" : (320, 320),
        "gpu_num" : 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        class_names = ["green","white","yellow"]
        return class_names

    def _get_anchors(self):
        return np.array([[10.,13.],[16.,30.],[ 33., 23.],[ 30.,  61.],[ 62.,  45.],[ 59., 119.],[116.,  90.],[156., 198.],[373. ,326.]])


    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes


    def detect_fast(self,image_size,image_expand):
        '''
        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
        '''
        
        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_expand,
                self.input_image_shape: [image_size[1], image_size[0]],
                K.learning_phase(): 0
            })
        
        return out_boxes,out_scores,out_classes

    
    
    def detect_image(self, image):
        start = timer()

        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
        
        
        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })
        
        return out_boxes,out_scores,out_classes
    
    
    
    
        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        end = timer()
        print(end - start)
        return image

    def close_session(self):
        self.sess.close()

def get_ground_truth_annotations(image_path):
    """Reads the ground_thruth information from either the tablet annotations (imagename_annotations.json),
        the LabelMe annotations (imagename.json) or tensorflow xml format annotations (imagename.xml)

    Parameters:
        image_path (str): path to the image of which the annotations should be read
    
    Returns:
        list: a list containing all annotations corresponding to that image.
            Returns the None if no annotation file is present
    """
    ground_truth = file_utils.get_annotations(image_path)
    if len(ground_truth) == 0:                     
        return None
    return ground_truth



if __name__== "__main__":
    
    model_path = constants.color_model_path
    images_to_predict = constants.project_folder + "/images/train"
    images_to_predict = "C:/Users/johan/Desktop/analysis/Test4_2.mp4/detected_bees"
    output_folder = constants.predictions_folder
    output_folder = "C:/Users/johan/Desktop/test"
    classes = ["green","white","yellow"]
    
    #predict(model_path,images_to_predict,output_folder,classes)