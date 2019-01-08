#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 06:34:48 2018

@author: ramanathan
"""

import pandas as pd
import tensorflow as tf
import numpy as np
import cv2
from predict_image import load_graph, load_labels, read_tensor_from_image_file
#from video_stream import VideoStream

def predict():
    
      model_file = "./v2/CU36_CU110/inception/retrained_graph.pb"
      label_file = "./v2/CU36_CU110/inception/retrained_labels.txt"
      input_layer = "Placeholder"
      output_layer = "final_result"
      frame_count = 0
    
      font=cv2.FONT_HERSHEY_TRIPLEX
      font_color=(255,255,255)

    
      graph = load_graph(model_file)
      input_name = "import/" + input_layer
      output_name = "import/" + output_layer
      vs = cv2.VideoCapture("../data/input/v2/CU36_CU110/20180910_033055_154/2.mp4")
      fps = vs.get(cv2.CAP_PROP_FPS)
      out = cv2.VideoWriter("../data/output/v2/results/CU36_CU110/20180910_033055_154_2_analysis.avi",
                                  cv2.VideoWriter_fourcc('D','I','V','X'), fps, (640,480))
      actions_df = pd.DataFrame(columns = ['Action','Probability'])
      actions_df.index.name = 'Frame'
      with tf.Session(graph=graph) as sess:
          input_operation = graph.get_operation_by_name(input_name)
          output_operation = graph.get_operation_by_name(output_name)
          while True:
              grabbed, frame = vs.read()
              if grabbed is False:
                  break
              frame_count += 1
              
              if frame_count % 1 == 0:
                  
                  cv2.imwrite("current_frame.jpg",frame)
                  t = read_tensor_from_image_file("./current_frame.jpg")
                  
                  results = sess.run(output_operation.outputs[0], {input_operation.outputs[0]: t})
                  results = np.squeeze(results)
    
                  top_k = results.argsort()[-1:][::-1]
                  labels = load_labels(label_file)
                  for i in top_k:
                      score = results[i]
                      prediction = labels[i]
                  if frame_count>10:
                        # Change the text position depending on your camera resolution
                        cv2.putText(frame,prediction, (20,400),font, 1, font_color)
                        cv2.putText(frame,"{:.2f}%".format(np.round(score*100,2)),(20,440),font,1,font_color)
                        actions_df.loc[frame_count] = ([prediction, score])
                  out.write(frame)
                  cv2.imshow("Frame", frame)
                  if cv2.waitKey(1) & 0xFF == 27:
                      break
    
      vs.release()  
      out.release()
      cv2.destroyAllWindows() 
      sess.close()
      actions_df.to_csv("../data/output/v2/results/CU36_CU110/20180910_033055_154_2_analysis.csv")
      
if __name__ == "__main__":
      predict()      
