import os
import numpy as np
import tensorflow
from PIL import Image

IMAGE_SIZE = 299
IMAGE_CHANNELS = 3


def image_classification(image_path):
    try:
        model = tensorflow.keras.applications.inception_v3.InceptionV3(weights='imagenet')
        tf_default_graph = tensorflow.get_default_graph()
        pil_img = Image.open(image_path)
        pil_img = pil_img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
        pil_img = np.array(pil_img)

        if len(pil_img.shape) == 2:
            pil_img = pil_img.reshape((1, IMAGE_SIZE, IMAGE_SIZE, 1))
            pil_img = np.repeat(pil_img, 3, axis=3)

        pil_img = pil_img.reshape((1, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS))

        with tf_default_graph.as_default():
            x = tensorflow.keras.applications.inception_v3.preprocess_input(pil_img)
            y_hat = model.predict(x)
            top3 = tensorflow.keras.applications.inception_v3.decode_predictions(y_hat, top=3)[0]
            names = list(map(lambda e: e[1], top3))
            probs = list(map(lambda e: str(round(e[2] * 100, 1)) + "%", top3))
            output = {}
            for i in range(0, len(names)):
                output[names[i]] = probs[i]
        return output
    finally:
        os.remove(image_path)

