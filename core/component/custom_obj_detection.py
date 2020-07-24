from shutil import copyfile

from PIL import Image

from core.lib.custom_image_classification import extract_data
from core.utils.storage import file_download
from resources.object_detection.utils import dataset_util
from collections import namedtuple
import io
import os
import pandas as pd
import tensorflow as tf
from google.protobuf import text_format
from resources.object_detection.protos import pipeline_pb2


def label_map(classes, OBJ_DIR):
    end = '\n'
    s = ' '
    out = ''
    for ID, name in enumerate(classes):
        out += ''
        out += 'item' + s + '{' + end
        out += s * 2 + 'id:' + ' ' + (str(ID + 1)) + end
        out += s * 2 + 'name:' + ' ' + '\'' + str(name) + '\'' + end
        out += '}' + end * 2

    with open(OBJ_DIR+"/label_map.pbtxt", 'a') as f:
        f.write(out)


def update_ssd_inception_config_file(OBJ_DIR, classes, name, steps, worker):
    pipeline = "resources/ssd_inception_v2_coco.config"
    output = OBJ_DIR+"/ssd_inception_v2_coco.config"

    os.mkdir(name+"_"+worker+"_cod_model")
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()

    with tf.gfile.GFile(pipeline, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, pipeline_config)

    pipeline_config.model.ssd.num_classes = len(classes)
    pipeline_config.train_config.num_steps = steps

    pipeline_config.train_input_reader.label_map_path = OBJ_DIR+"/label_map.pbtxt"
    pipeline_config.train_input_reader.tf_record_input_reader.input_path[0] = OBJ_DIR+"/labels.record"

    # pipeline_config.eval_input_reader.label_map_path = OBJ_DIR+"/eval_label_map.pbtxt"
    # pipeline_config.train_input_reader.tf_record_input_reader.input_path[1] = OBJ_DIR + "/eval_labels.record"

    config_text = text_format.MessageToString(pipeline_config)
    with tf.gfile.Open(output, "wb") as f:
        f.write(config_text)


def generate_tfrecords(train_request, user_id, worker):
    zip_file_url = train_request.zip_file_url
    images_column = train_request.images
    labels_column = train_request.labels
    name = train_request.name
    steps = train_request.steps

    zip_path = file_download(zip_file_url, user_id)
    images_folder = extract_data(zip_path, worker)
    csv_file = ""
    for i in os.listdir(images_folder):
        if i.endswith(".csv"):
            csv_file = i
            break
    images_df = pd.read_csv(images_folder + "/" + csv_file)

    content_list = []
    images = images_df[images_column]
    labels = images_df[labels_column]

    classes = images_df[labels_column].unique()
    for i in range(0, len(images)):
        if not images[i].endswith('.csv'):
            file =images_folder + "/" + images[i]
            im = Image.open(file)
            width, height = im.size
            value = (file, width, height, labels[i], 0, 0, width, height)
            content_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(content_list, columns=column_name)

    image_df = xml_df

    os.mkdir(worker+"_custom_clf")
    OBJ_DIR = worker+"_custom_clf"

    image_df.to_csv((OBJ_DIR + "/" + 'data.csv'), index=None)

    labels = images_df[labels_column].unique()
    label_to_index = dict((name, index) for index, name in enumerate(labels))

    def class_text_to_int(row_label):
        index = label_to_index.get(row_label)
        return index

    def split(df, group):
        data = namedtuple('data', ['filename', 'object'])
        gb = df.groupby(group)
        return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]

    def create_tf_example(group, path):
        with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
            encoded_jpg = fid.read()
        encoded_jpg_io = io.BytesIO(encoded_jpg)
        image = Image.open(encoded_jpg_io)
        width, height = image.size

        filename = group.filename.encode('utf8')
        image_format = b'jpg'
        xmins = []
        xmaxs = []
        ymins = []
        ymaxs = []
        classes_text = []
        classes = []

        for index, row in group.object.iterrows():
            xmins.append(row['xmin'] / width)
            xmaxs.append(row['xmax'] / width)
            ymins.append(row['ymin'] / height)
            ymaxs.append(row['ymax'] / height)
            classes_text.append(row['class'].encode('utf8'))
            classes.append(class_text_to_int(row['class']))

        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': dataset_util.int64_feature(height),
            'image/width': dataset_util.int64_feature(width),
            'image/filename': dataset_util.bytes_feature(filename),
            'image/source_id': dataset_util.bytes_feature(filename),
            'image/encoded': dataset_util.bytes_feature(encoded_jpg),
            'image/format': dataset_util.bytes_feature(image_format),
            'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
            'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
            'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
            'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
            'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
            'image/object/class/label': dataset_util.int64_list_feature(classes),
        }))
        return tf_example

    csv_input = OBJ_DIR + "/data.csv"
    output_path = OBJ_DIR+ "/labels.record"
    img_path = ""

    writer = tf.python_io.TFRecordWriter(output_path)
    path = os.path.join(img_path)
    examples = pd.read_csv(csv_input)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())

    writer.close()
    print('Successfully created the TFRecords: {}'.format(output_path))

    print(classes)
    label_map(classes, OBJ_DIR)
    print("successfully created label_map.pbtxt")

    update_ssd_inception_config_file(OBJ_DIR, classes, name, steps, worker)
    print("Successfully created config file")
    copyfile(OBJ_DIR + "/label_map.pbtxt", name+"_"+worker+"_cod_model" + "/label_map.pbtxt")

    return images_folder

