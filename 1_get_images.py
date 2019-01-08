#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 17:34:33 2018

@author: ramanathan
"""
import os
import cv2
import sys

def get_images_from_video(input_file, save_folder):


    # Capture frames
    cap = cv2.VideoCapture(input_file)
    frame_exists, image = cap.read()
    #frame_count = 0
    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)

    while frame_exists:
        cv2.imwrite(os.path.join(save_folder, "{}.jpg".format(timestamp)), image)
        frame_exists, image = cap.read()
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
        #frame_count += 1

if __name__ == '__main__':
    camera_id = sys.argv[1]
    for root, dirs, files in os.walk(os.path.join("../data/input/v2/"), camera_id):
        for name in files:
            save_folder = os.path.join("../data/output/v2/frames/", camera_id, os.path.split(root)[1] + "_" + os.path.splitext(name)[0])
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            get_images_from_video(os.path.join(root,name), save_folder)