from __future__ import absolute_import, division
import json
from shutil import rmtree

import keras
import numpy as np
import pandas as pd
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import load_model, Model
from numpy.distutils.system_info import NotFoundError
from sklearn.model_selection import train_test_split
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.applications.inception_v3 import InceptionV3, preprocess_input

from core.component.file_component import *
from core.utils.storage import *
import tensorflow as tf

WIDTH = 299
HEIGHT = 299
BATCH_SIZE = 32
VALIDATION_STEPS = 64


def set_up_model(num_classes):
    base_model = InceptionV3(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dropout(0.4)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in base_model.layers:
        layer.trainable = False
    return model


def data_prep():
    datagen = ImageDataGenerator(preprocessing_function=preprocess_input, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2,
                                 shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
    return datagen


def custom_image_classification(train_request, user_id, worker):
    worker = str(worker)
    zip_file_url = train_request.zip_file_url
    images_column = train_request.images
    labels_column = train_request.labels
    train_percentage = train_request.train_percentage
    name = train_request.name
    storage_label = train_request.label
    epochs = train_request.epochs
    steps_per_epoch = train_request.steps_per_epoch
    train_dir = "train" + worker
    validation_dir = "validation" + worker
    model_folder = "saved_model_" + worker
    model_file = model_folder + "/" + name + '.model'
    model_zip_file = model_folder + ".zip"
    model_json = model_folder + "/model.json"

    os.mkdir(train_dir)
    os.mkdir(validation_dir)
    os.mkdir(model_folder)
    try:
        zip_path = file_download(zip_file_url, user_id)
        images_folder = extract_data(zip_path, worker)

        csv_file = ""
        for i in os.listdir(images_folder):
            if i.endswith(".csv"):
                csv_file = i
                break


        images = pd.read_csv(images_folder + "/" + csv_file)
        train, test = train_test_split(images, train_size=train_percentage/100, random_state=40)
        num_classes = len(images[labels_column].unique())

        for img in train[images_column]:
            img_path = images_folder + "/" + img
            shutil.copy(img_path, train_dir)
        for img in test[images_column]:
            img_path = images_folder + "/" + img
            shutil.copy(img_path, validation_dir)

        model = set_up_model(num_classes)
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

        for img in train.values:
            filename = img[0]
            label = img[1]
            src = os.path.join("./" + train_dir, filename)
            label_dir = os.path.join("./" + train_dir, label)
            dest = os.path.join(label_dir, filename)
            im = Image.open(src)
            rgb_im = im.convert('RGB')
            if not os.path.exists(label_dir):
                os.makedirs(label_dir)
            rgb_im.save(dest)
            os.remove(src)

        for img in test.values:
            filename = img[0]
            label = img[1]
            src = os.path.join("./", validation_dir, filename)
            label_dir = os.path.join("./", validation_dir, label)
            dest = os.path.join(label_dir, filename)
            im = Image.open(src)
            rgb_im = im.convert('RGB')
            if not os.path.exists(label_dir):
                os.makedirs(label_dir)
            rgb_im.save(dest)
            os.remove(src)

        for i in images[labels_column].unique():
            if not os.path.exists(os.path.join("./", train_dir, i)):
                os.makedirs(os.path.join("./", train_dir, i))
        for i in images[labels_column].unique():
            if not os.path.exists(os.path.join("./", validation_dir, i)):
                os.makedirs(os.path.join("./", validation_dir, i))

        datagen = data_prep()
        train_generator = datagen.flow_from_directory("./train"+worker+"/", target_size=(HEIGHT, WIDTH), batch_size=BATCH_SIZE, class_mode='categorical')
        validation_generator = datagen.flow_from_directory("./validation"+worker+"/", target_size=(HEIGHT, WIDTH), batch_size=BATCH_SIZE, class_mode='categorical')

        print(train_generator.class_indices)


        x = model.fit_generator(train_generator, epochs=epochs, steps_per_epoch=steps_per_epoch, validation_data=validation_generator, validation_steps=VALIDATION_STEPS)

        labels_config = {}
        j = 0
        for cls in train_generator.class_indices:
            labels_config[j] = cls
            j += 1
        print(labels_config)
        model.save(model_file)
        loss = x.history.get('loss')[-1]
        accuracy = x.history.get('acc')[-1]

        with open(model_json, "w") as f:
            f.write(json.dumps(labels_config))

        file_paths = []
        for root, directories, files in os.walk(model_folder):
            for filename in files:
                filepath = os.path.join(root, filename)
                file_paths.append(filepath)

        with ZipFile(model_zip_file, 'w') as model_zip:
            for file in file_paths:
                model_zip.write(file)
        model_file_url = file_upload(model_zip_file, user_id, storage_label)
        return model_file_url, loss, accuracy

    finally:
        rmtree(train_dir)
        rmtree(validation_dir)
        rmtree(images_folder)
        rmtree(model_folder)
        os.remove(model_zip_file)


def prediction_zip(prediction_request, user_id, worker):
    worker = str(worker)
    model_file_url = prediction_request.model_url
    test_zip_folder = prediction_request.zip_file_url
    storage_label = prediction_request.label
    name = prediction_request.name

    zip_path = file_download(test_zip_folder, user_id)
    images_folder = extract_data(zip_path, worker)

    zip_path = file_download(model_file_url, user_id)
    model_folder = extract_data(zip_path, worker)

    model_file = model_json = ""
    for i in os.listdir(model_folder):
        if i.endswith(".model"):
            model_file = i
        if i.endswith(".json"):
            model_json = i
    if model_file is None:
        raise NotFoundError("model not found")
    if model_json is None:
        raise NotFoundError("model json not found")

    with open(model_folder+"/"+model_json) as f:
        class_config = json.load(f)

    pred_file = name + "_" + worker + "_prediction.csv"
    model = load_model(model_folder + "/" + model_file)
    images = os.listdir(images_folder)

    try:
        labels = []
        imgs = []
        classes = []
        for img_name in images:
            if not img_name.endswith(".csv"):
                path = images_folder+"/"+img_name
                img = image.load_img(path, target_size=(HEIGHT, WIDTH))
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)
                preds = model.predict(x)[0]
                result = {}
                z = 0
                for i in class_config.keys():
                    label = class_config[i]
                    result[label] = preds[z]
                    z += 1
                print(result)
                cls = sorted(result.items(), key=lambda j: j[1], reverse=True)[0][0]
                imgs.append(img_name)
                labels.append(result)
                classes.append(cls)
        df = pd.DataFrame(list(zip(imgs, labels, classes)), columns=["image_name", "prediction_json", "prediction_class"])
        df.to_csv(pred_file, index=False)
        csv_file_url = file_upload(pred_file, user_id, storage_label)
        return csv_file_url
    finally:
        rmtree(model_folder)
        os.remove(pred_file)
        rmtree(images_folder)


def prediction_image(image_path, model_url, user_id):
    try:
        model_folder = extract_model(model_url, user_id)

        model_file = model_json = ""
        for i in os.listdir(model_folder):
            if i.endswith(".model"):
                model_file = i
            if i.endswith(".json"):
                model_json = i
        if model_file is None:
            raise NotFoundError("model not found")
        if model_json is None:
            raise NotFoundError("model json not found")

        with open(model_folder+"/"+model_json) as f:
            class_config = json.load(f)
        print(class_config)
        model = load_model(model_folder + "/" + model_file)

        img = image.load_img(image_path, target_size=(HEIGHT, WIDTH))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        pred = model.predict(x)[0]
        result = {}
        z = 0
        for i in class_config.keys():
            label = class_config[i]
            result[label] = str(pred[z])
            z += 1
        print(result)

        keras.backend.clear_session()
        return result
    finally:
        os.remove(image_path)
