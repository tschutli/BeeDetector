# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 17:46:20 2020

@author: johan
"""

from utils import file_utils
import shutil
import os
import progressbar

def extract_annotated_images(input_folder,output_folder):
    all_images = file_utils.get_all_image_paths_in_folder(input_folder)
    if not all_images:
        print("No input images found. Please check the provided input path")
        return
    print("Checking all input images, whether they have annotations or not...", flush=True)
    count = 0
    for image_path in progressbar.progressbar(all_images):
        src_annotation_path = image_path[:-4] + ".xml"
        annotations = file_utils.get_annotations_from_xml(src_annotation_path)
        if len(annotations) > 0:
            count += 1
            dest_image_path = os.path.join(output_folder,os.path.basename(image_path))
            dest_annotation_path = os.path.join(output_folder, os.path.basename(src_annotation_path))
            shutil.copyfile(image_path,dest_image_path)
            shutil.copyfile(src_annotation_path,dest_annotation_path)
    print("Done. Copied " + str(count) + " images and annotation files.")

if __name__ == '__main__':
    input_folder = "C:/Users/johan/Downloads/detected_bees_vid4_5/detected_bees"
    output_folder = "C:/Users/johan/Desktop/Data/video4/numbers5"
    
    extract_annotated_images(input_folder,output_folder)
