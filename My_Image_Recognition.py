from picamera.array import PiRGBArray
from picamera import PiCamera
from utils import label_map_util
from utils import visualization_utils as vis_util
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import numpy as np

Image_Width = 1280

Image_Height = 720

sys.path.append('..')

model_name = 'ssdlite_mobilenet_v2_coco_2018_05_09'

cwd_path = os.getcwd()

path_to_ckpt = os.path.join(cwd_path,model_name,'frozen_inference_graph.pb')

path_to_labels = os.path.join(cwd_path,'data','mscoco_label_map.pbtxt')

num_classes = 90

label_map = label_map_util.load_labelmap(path_to_labels)

categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_classes, use_display_name=True)

category_index = label_map_util.create_category_index(categories)

#Load Tensor Flow
detection_graph = tf.Graph()

with detection_graph.as_default():

    od_graph_def = tf.GraphDef()

    with tf.gfile.GFile(path_to_ckpt,'rb') as fid:

        serialized_graph = fid.read()

        od_graph_def.ParseFromString(serialized_graph)

        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')

detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

num_detections = detection_graph.get_tensor_by_name('num_detections:0')

#Object Detection
camera = PiCamera()
camera.resolution = (Image_Width,Image_Height)
camera.framerate = 10
imgplot = plt.imshow(camera.capture())
        

