#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 17:52:57 2018

@author: ramanathan
"""

import pickle
import sys
import os
import tensorflow as tf
from tqdm import tqdm
from predict_image import load_graph, load_labels, read_tensor_from_image_file

camera_id = 'CU36_CU110'

def predict_on_frames(camera_id, model_file, frames, batch):
    
#    input_height = 299
#    input_width = 299
#    input_mean = 0
#    input_std = 255
    input_layer = "Placeholder"
    output_layer = "final_result"
    
    frame_count = 0
    graph = load_graph(model_file)
    
    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    
    with tf.Session(graph=graph) as sess:
        input_operation = graph.get_operation_by_name(input_name)
        output_operation = graph.get_operation_by_name(output_name)

        frame_predictions = []
        image_path = '../data/output/v2/frames/' + camera_id + '/' + batch + '/'
        pbar = tqdm(total=len(frames))
        for i, frame in enumerate(frames):
            
            filename = frame[0]
            label = frame[1]


    		# Get the image path.
            file_name = image_path + filename + '.jpg'
    
    		# Read in the image_data
            t = read_tensor_from_image_file(file_name)
    
            try:
                predictions = sess.run(output_operation.outputs[0],
    		                           {input_operation.outputs[0]: t})
                prediction = predictions[0]
            except KeyboardInterrupt:
                print("You quit with ctrl+c")
                sys.exit()
            except:
                print("Error making prediction, continuing.")
                continue
    
    		# Save the probability that it's each of our classes.
            frame_predictions.append([prediction, label])
    
            if i > 0 and i % 10 == 0:
                pbar.update(10)
    
        pbar.close()

        return frame_predictions

def get_accuracy(predictions, labels):
    correct = 0
    for frame in predictions:
        # Get the highest confidence class.
        this_prediction = frame[0].tolist()
        this_label = frame[1]

        max_value = max(this_prediction)
        max_index = this_prediction.index(max_value)
        predicted_label = labels[max_index]

        # Now see if it matches
        if predicted_label == this_label:
            correct += 1

    accuracy = correct / len(predictions)
    return accuracy

def main():
    batches = ['20180910_033055_154_1']
    model_file = "./v2/CU36_CU110/inception/retrained_graph.pb"
    label_file = "./v2/CU36_CU110/inception/retrained_labels.txt"
    labels = load_labels(label_file)

    for batch in batches:
        print("Doing batch %s" % batch)
        with open(os.path.join('../data/output/v2/labelled_frames', camera_id, str(batch) + ".pkl"), 'rb') as fin:
            frames = pickle.load(fin)

        # Predict on this batch and get the accuracy
        predictions = predict_on_frames(camera_id, model_file, frames, batch)
        accuracy = get_accuracy(predictions, labels)
        print("Batch accuracy: %.5f" % accuracy)

        # Save it
        with open(os.path.join('../data/output/v2/predicted_frames', camera_id, str(batch) + '.pkl'), 'wb') as fout:
            pickle.dump(predictions, fout)

    print("Done")

if __name__ == '__main__':
    main()
