#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 19:40:45 2018

@author: ramanathan
"""
import glob
import pickle
import os
from shutil import copyfile
import sys
camera_id = 'CU36_CU110'
from actionsv2_CU36_CU110 import actions

def label_frames(camera_id, batch, actions, copyimage=True):
    # Get all the frames
    frames = sorted(glob.glob(os.path.join('../data/output/v2/frames', camera_id, str(batch), '*')),
                    key = lambda file_name: float(file_name.split("/")[-1][:-4]))
    
    n_frames = 0

    # Get timestamp end for training
    train_timestamp_end = 330000.0

    #print("Labelling {:d} frames".format(n_frames))
    print("Labelling frames")                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
    
    # Loop through the frames
    labelled_frames = []
    n_undefined = 0
    
    #Make dirs for labels
    if copyimage:
        if not os.path.exists(os.path.join('../data/output/v2/classifications', camera_id)):
            for action in actions.keys():
                os.makedirs(os.path.join('../data/output/v2/classifications', camera_id, action))
            os.makedirs(os.path.join('../data/output/v2/classifications', camera_id, 'undefined'))

    for frame in frames:
        n_frames += 1
        # Get the timestamp in ms
        timestamp = frame.replace('.jpg', '').split('/')[-1]
        
        if float(timestamp) <= train_timestamp_end:

            # Get the action for the timestamp
            label = get_label(timestamp, actions)

            # Save the labelled frames 
            labelled_frames.append([timestamp, label])

            # Copy the labelled frames
            if copyimage:
                copyfile(frame, os.path.join('../data/output/v2/classifications', camera_id, label, str(batch)) + '_' + timestamp + '.jpg')

            # Count undefined frames
            if label == 'undefined':
                n_undefined += 1

        else:
            break    

    print("Labelling completed, with {:d} undefined frames and {:d} action frames".format(
            n_undefined, n_frames - n_undefined))

    with open(os.path.join('../data/output/v2/labelled_frames', camera_id, str(batch) + ".pkl"), 'wb') as fout:
        pickle.dump(labelled_frames, fout)

    return labelled_frames



def get_label(timestamp, action):
    
    for action_type, action_timeframes in action.items():
        for action_t in action_timeframes:
            if float(action_t['start']) <= float(timestamp) <= float(action_t['end']):
                return action_type
    return 'undefined'

if __name__ == '__main__':
    '''
    camera_id = sys.argv[1]
    batches = []
    extracted_batches = []
    for root, dirs, file in os.walk(os.path.join("../data/output/v2/frames"), camera_id):
        for name in dirs:
            extracted_batches.append(name)
    for root, dirs, files in os.walk(os.path.join("../data/input/v2", camera_id)):       
        for name in files:
            batch_name = os.path.split(root)[1] + "_" + os.path.splitext(name)[0]
            if batch_name in extracted_batches:
                batches.append(batch_name)
    print("Will attach labels for the batches: ", batches)
    '''
    batches = ['20180910_033055_154_1']
    for batch in batches:
        label_frames(camera_id, batch, actions[str(batch)], copyimage=True)
