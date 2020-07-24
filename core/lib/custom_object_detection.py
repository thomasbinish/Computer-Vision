import base64
import functools
import json
import os
import random
from shutil import rmtree
from zipfile import ZipFile

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from core.component.custom_obj_detection import generate_tfrecords
from core.component.file_component import *
from core.utils.storage import file_upload, file_download
from resources.object_detection.builders import dataset_builder, graph_rewriter_builder
from resources.object_detection.builders import model_builder
from resources.object_detection.legacy import trainer
from resources.object_detection.utils import config_util
import tensorflow as tf
from google.protobuf import text_format
from resources.object_detection import exporter
from resources.object_detection.protos import pipeline_pb2
from resources.object_detection.utils import visualization_utils as vis_util, ops as utils_ops, label_map_util


def custom_obj_detection_train(train_request, user_id, worker_):
    worker_ = str(worker_)
    name = train_request.name
    storage_label = train_request.label
    steps = train_request.steps
    images_folder = generate_tfrecords(train_request, user_id, worker_)
    file_name = name + "_" + worker_ + "_cod_model"

    task = 0
    num_clones = 1
    clone_on_cpu = False

    OBJ_DIR = worker_ + "_custom_clf"
    train_dir = OBJ_DIR
    pipeline_config_path = OBJ_DIR + "/ssd_inception_v2_coco.config"
    train_config_path = ''
    input_config_path = ''
    model_config_path = ''
    try:

        if task == 0:
            tf.gfile.MakeDirs(train_dir)
        if pipeline_config_path:
            configs = config_util.get_configs_from_pipeline_file(
                pipeline_config_path)
            if task == 0:
                tf.gfile.Copy(pipeline_config_path,
                              os.path.join(train_dir, 'pipeline.config'),
                              overwrite=True)
        else:
            configs = config_util.get_configs_from_multiple_files(
                model_config_path=model_config_path,
                train_config_path=train_config_path,
                train_input_config_path=input_config_path)
            if task == 0:
                for name, config in [('model.config', model_config_path),
                                     ('train.config', train_config_path),
                                     ('input.config', input_config_path)]:
                    tf.gfile.Copy(config, os.path.join(train_dir, name),
                                  overwrite=True)

        model_config = configs['model']
        train_config = configs['train_config']
        input_config = configs['train_input_config']

        model_fn = functools.partial(
            model_builder.build,
            model_config=model_config,
            is_training=True)

        def get_next(config):
            return dataset_builder.make_initializable_iterator(dataset_builder.build(config)).get_next()

        create_input_dict_fn = functools.partial(get_next, input_config)

        env = json.loads(os.environ.get('TF_CONFIG', '{}'))
        cluster_data = env.get('cluster', None)
        cluster = tf.train.ClusterSpec(cluster_data) if cluster_data else None
        task_data = env.get('task', None) or {'type': 'master', 'index': 0}
        task_info = type('TaskSpec', (object,), task_data)

        # Parameters for a single worker.
        ps_tasks = 0
        worker_replicas = 1
        worker_job_name = 'lonely_worker'
        task = 0
        is_chief = True
        master = ''

        if cluster_data and 'worker' in cluster_data:
            worker_replicas = len(cluster_data['worker']) + 1
        if cluster_data and 'ps' in cluster_data:
            ps_tasks = len(cluster_data['ps'])

        if ps_tasks < 1 < worker_replicas:
            raise ValueError('At least 1 ps task is needed for distributed training.')

        if worker_replicas >= 1 and ps_tasks > 0:
            server = tf.train.Server(tf.train.ClusterSpec(cluster), protocol='grpc',
                                     job_name=task_info.type,
                                     task_index=task_info.index)
            if task_info.type == 'ps':
                server.join()
                return

            worker_job_name = '%s/task:%d' % (task_info.type, task_info.index)
            task = task_info.index
            is_chief = (task_info.type == 'master')
            master = server.target

        graph_rewriter_fn = None
        if 'graph_rewriter_config' in configs:
            graph_rewriter_fn = graph_rewriter_builder.build(
                configs['graph_rewriter_config'], is_training=True)

        trainer.train(
            create_input_dict_fn,
            model_fn,
            train_config,
            master,
            task,
            num_clones,
            worker_replicas,
            clone_on_cpu,
            ps_tasks,
            worker_job_name,
            is_chief,
            train_dir,
            graph_hook_fn=graph_rewriter_fn)

        input_type = 'image_tensor'
        input_shape = None
        pipeline_config_path = OBJ_DIR + "/ssd_inception_v2_coco.config"
        trained_checkpoint_prefix = OBJ_DIR + "/model.ckpt-"+str(steps)
        output_directory = name+"_"+worker_+"_cod_model"
        write_inference_graph = True
        config_override = ''

        slim = tf.contrib.slim
        pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
        with tf.gfile.GFile(pipeline_config_path, 'r') as f:
            text_format.Merge(f.read(), pipeline_config)
        text_format.Merge(config_override, pipeline_config)
        if input_shape:
            input_shape = [
                int(dim) if dim != '-1' else None
                for dim in input_shape.split(',')
            ]
        else:
            input_shape = None
        exporter.export_inference_graph(
            input_type, pipeline_config, trained_checkpoint_prefix,
            output_directory, input_shape=input_shape,
            write_inference_graph=write_inference_graph)

        file_paths = []
        for root, directories, files in os.walk(file_name):
            for filename in files:
                filepath = os.path.join(root, filename)
                file_paths.append(filepath)

        with ZipFile(file_name+'.zip', 'w') as zip:
            for file in file_paths:
                zip.write(file)

        file_url = file_upload(file_name+".zip", user_id, storage_label)
        return file_url

    finally:
        os.remove(file_name+".zip")
        rmtree(name + "_" + worker_ + "_cod_model")
        rmtree(OBJ_DIR)
        rmtree(images_folder)


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


def custom_obj_detection_predict(prediction_request, user_id, worker):
    images_zip_url = prediction_request.zip_file_url
    model_url = prediction_request.model_url
    name = prediction_request.name
    storage_label = prediction_request.label

    zip_path = file_download(images_zip_url, user_id)
    images_folder = extract_data(zip_path, worker)

    zip_path = file_download(model_url, user_id)
    model_folder = extract_data(zip_path, worker)


    try:
        MODEL_NAME = model_folder
        PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'
        PATH_TO_LABELS = MODEL_NAME + '/label_map.pbtxt'
        category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')


        def object_detection_json(image_path):
            image = Image.open(image_path)
            width, height = image.size
            image_np = load_image_into_numpy_array(image)
            image_np_expanded = np.expand_dims(image_np, axis=0)
            output_dict = run_inference_for_single_image(image_np_expanded, detection_graph)
            output = vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                output_dict['detection_boxes'],
                output_dict['detection_classes'],
                output_dict['detection_scores'],
                category_index,
                instance_masks=output_dict.get('detection_masks'),
                use_normalized_coordinates=True,
                line_thickness=2)
            x = 0
            response = {}

            for i in output["box_to_display_str_map"]:
                value = output["box_to_display_str_map"][i]
                name = value[0].split(":")[0]
                obj = "object_" + str(x)
                response[obj] = {
                    "name": name,
                    "accuracy": value[0].split(":")[1],
                    "coordinates": {
                        "point_0": [i[1] * width, i[0] * height],
                        "point_1": [i[3] * width, i[2] * height]
                    }
                }
                x += 1
            return response


        images = []
        outputs = []
        for file in os.listdir(images_folder):
            if not file.endswith(".csv"):
                output = object_detection_json(images_folder+"/"+file)
                outputs.append(output)
                images.append(file)

        df = pd.DataFrame(list(zip(images, outputs)), columns=["image_path", "prediction"])
        df.to_csv(name+".csv", index=False)
        file_url = file_upload(name+".csv", user_id, storage_label)
        return file_url
    finally:
        rmtree(images_folder)
        rmtree(model_folder)
        os.remove(name+".csv")


def custom_object_detection_image(model_url, image_path, user_id):
    model_folder = extract_model(model_url, user_id)
    MODEL_NAME = model_folder
    PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'
    PATH_TO_LABELS = MODEL_NAME + '/label_map.pbtxt'
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')


    image = Image.open(image_path)
    width, height = image.size
    image_np = load_image_into_numpy_array(image)
    image_np_expanded = np.expand_dims(image_np, axis=0)
    output_dict = run_inference_for_single_image(image_np_expanded, detection_graph)
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks'),
        use_normalized_coordinates=True,
        line_thickness=2)
    try:
        plt.figure(figsize=[width, height])
        plt.imsave(arr=image_np, fname="image")

        with open("image.png", "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        return encoded_string
    finally:
        os.remove("image.png")
        os.remove(image_path)


def custom_object_detection_json(model_url, image_path, user_id):
    model_folder = extract_model(model_url, user_id)
    MODEL_NAME = model_folder
    PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'
    PATH_TO_LABELS = MODEL_NAME + '/label_map.pbtxt'
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    try:

        image = Image.open(image_path)
        width, height = image.size
        image_np = load_image_into_numpy_array(image)
        image_np_expanded = np.expand_dims(image_np, axis=0)
        output_dict = run_inference_for_single_image(image_np_expanded, detection_graph)
        output = vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            instance_masks=output_dict.get('detection_masks'),
            use_normalized_coordinates=True,
            line_thickness=2)
        x = 0
        response = {}

        for i in output["box_to_display_str_map"]:
            value = output["box_to_display_str_map"][i]
            name = value[0].split(":")[0]
            obj = "object_" + str(x)
            response[obj] = {
                "name": name,
                "accuracy": value[0].split(":")[1],
                "coordinates": {
                    "point_0": [i[1] * width, i[0] * height],
                    "point_1": [i[3] * width, i[2] * height]
                }
            }
            x += 1
        return response
    finally:
        os.remove(image_path)