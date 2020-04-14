# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 11:42:59 2020

@author: johan
"""

def filter_annotations(annotations,labels):
    for annotation in annotations:
        #Remove all spaces and numbers from annotation name
        filtered_name = ''.join(x for x in annotation["name"] if x.isalpha())
        annotation["name"] = filtered_name
    

if __name__ == '__main__':
    annotations = [{"name":"abcd 334"}]
    filter_annotations(annotations,"")