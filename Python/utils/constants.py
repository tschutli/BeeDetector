# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 14:17:19 2019

@author: gallmanj
"""



'''MOST IMPORTANT SETTINGS'''
project_folder = "C:/Users/johan/Desktop/Agroscope_working_dir"
pretrained_model_link = "http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_2018_01_28.tar.gz"


'''image-preprocessing command parameters'''
input_folders = ["C:/Users/johan/Desktop/Agroscope/Data/Video1/bees"]

test_splits = [0.1]

validation_splits = [0.1]


#All images will be resized to tensorflow_tile_size x tensorflow_tile_size tiles
#choose a smaller tensorflow_tile_size if your gpu is not powerful enough to handle
#900 x 900 pixel tiles
tensorflow_tile_size = (320,240)


'''train command parameters'''
max_steps = 130000
with_validation = True
model_selection_criterion = "f1" #also used for export-inference-graph command

'''predict command parameters'''
images_to_predict = project_folder + "/images/test_full_size"
predictions_folder = project_folder + "/predictions"
prediction_tile_size = 450
prediction_overlap = 50
min_confidence_score = 0.2 #also used by evaluate command
visualize_predictions = True
visualize_groundtruth = False
visualize_name = False
visualize_score = False
max_iou = 0.3


'''evaluate cmomand parameters'''
prediction_evaluation_folder = predictions_folder + "/evaluations"
iou_threshold = 0.3


'''visualization command parameters'''
visualize_bounding_boxes_with_name = True
clean_output_folder = True

'''copy-annotations command parameters'''
one_by_one = True

'''prepare-for-tablet command parameters'''
prepare_for_tablet_tile_size = 256


'''generate-heatmap command parameters'''
heatmap_width=100
max_val = None
classes = None
overlay = True
output_image_width = 1000
