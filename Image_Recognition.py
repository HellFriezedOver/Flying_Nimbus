import os
import numpy as np
import picamer.array import PiRGBArray
from picamera import PiCamera
import tensorflow as tf
import argparse
import sys
import time
import keyboard

#Setting up the Camera Constants
IM_Width = 1280
IM_Height = 720

#Camera Type
camera_type = 'picamera'
parser = argparse.ArgumentParser()

args = parser.pars_args()

sys.path.append('..')

from utils import label_map_util
from utils import visualization_utils as vis_util


MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09'

CWD_PATH = os.getcwd()

PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph')

PATH_TO_LABELS = os.path.join(CWD_PATH,'data','mscoco_label_map.pbtxt')

NUM_CLASSES = 90

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map,max_num_classes=NUM_CLASSES
category_index = label_map_util.create_category_index(categories)

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def=tf.GraphDef()
    with tf.gfile.Gfile(PATH_TO_CKPT,'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def,name='')
    sess = tf.Session(graph=detection_graph)

image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')

detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

frame_rate_calc = 1

#freq = getTickFrequency() on matplotlib
freq = 1500000000

                                                            
#font = how to set font on matplotlib

camera = PiCamera()
camera.resolution = (IM_Width,IM_Height)
camera.framerate = 10
rawCapture = PiRGBArray(camera,size=(IM_Width,IM_Height))
rawCapture.truncate(0)

for frame1 in camera.capture_continuous(rawCapture,format="bgr",use_video_port=True):
    t1 = time.clock()
    frame = np.copy(frame1.array)
    frame.setflags(write=1)                                                        
    frame_expanded = np.expand_dims(frame, axis=0)

    (boxes,scores, classes, num) = sess.run(
        [detection_boxes,detection_scores,detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})
    vis_util.visualize_boxes_and_labels_on_image_array(
        frame,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8
        min_score_thresh=0.40)
    t2= time.clock
    time1=(t2-t1)/freq
    frame_rate_calc = 1/time1
    if keyboard.is_pressed('q'):
        break                                                        
    rawCapture.truncate(0)                                                        
camera.close()






