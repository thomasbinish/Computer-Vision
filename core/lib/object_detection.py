import base64
import numpy as np
import os
import tensorflow as tf
from PIL import Image
from resources.object_detection.utils import visualization_utils as vis_util, ops as utils_ops, label_map_util
import matplotlib.pyplot as plt
from imageai.Detection import ObjectDetection
import os
import cv2
from keras import backend as K
MODEL_NAME = 'resources/object_detection/ssd_mobilenet_v1_coco_2017_11_17'
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('resources/object_detection/data', 'mscoco_label_map.pbtxt')



detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


def load_image_into_numpy_array(image_):
    (im_width, im_height) = image_.size
    return np.array(image_.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes', 'detection_masks']:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
            if 'detection_masks' in tensor_dict:
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(detection_masks, detection_boxes, image.shape[1], image.shape[2])
                detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            output_dict = sess.run(tensor_dict, feed_dict={image_tensor: image})
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.int64)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict


def object_detection_json(image_path):
    execution_path = os.getcwd()
    detector = ObjectDetection()
    detector.setModelTypeAsRetinaNet()
    detector.setModelPath( os.path.join(execution_path, "resources/resnet50_coco_best_v2.0.1.h5"))
    detector.loadModel()
    detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , image_path))
    print(type(detections))
    print(detections[0])
    dt={}
    for i in detections:
        if i["percentage_probability"] > 0:
            dt[i["percentage_probability"]]=[i["name"],i["box_points"]]
    K.clear_session()
    return dt




def object_detection_image(image_path):
    execution_path = os.getcwd()
    detector = ObjectDetection()
    detector.setModelTypeAsRetinaNet()
    detector.setModelPath( os.path.join(execution_path, "resources/resnet50_coco_best_v2.0.1.h5"))
    detector.loadModel()
    detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , image_path))
    print(type(detections))
    print(detections[0])
    dt={}
    for i in detections:
        if i["percentage_probability"] > 0:
            dt[i["percentage_probability"]]=[i["name"],i["box_points"]]
    ds =sorted (dt.keys())
    print(type(ds))
    ds=ds[(len(ds)-20):len(ds)]
    for i in ds:
        print(dt[i])


# path
    path = image_path

# Reading an image in default mode
    image = cv2.imread(path)

# Window name in which image is displayed
    window_name = 'Image'

# Start coordinate, here (5, 5)
# represents the top left corner of rectangle
    start_point = (5, 5)

# Ending coordinate, here (220, 220)
# represents the bottom right corner of rectangle
    end_point = (220, 220)

# Blue color in BGR
    color = (255, 0, 0)

# Line thickness of 2 px
    thickness = 2

# Using cv2.rectangle() method
# Draw a rectangle with blue line borders of thickness of 2 px
# Displaying the image
    for i in ds:
        b=(dt[i][1][0],dt[i][1][1])
        e=(dt[i][1][2],dt[i][1][3])
        cv2.rectangle(image, b, e, color, thickness)
    retval, buffer = cv2.imencode('.jpg', image)
    encoded_string = base64.b64encode(buffer)
    K.clear_session()
    return encoded_string
