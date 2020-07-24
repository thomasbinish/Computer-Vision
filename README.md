# DLTK-VISION-CORE

# Computer Vision
Computer Vision is a field of Artificial Intelligence that deals with understanding and implementing aspects of the Visual World. It applies deep learning models on Images and Videos to identify , interpret , classify and modify visual information.
DLTK Computer Vision Module provides the following APIs for now:
 - Face Detection Image / JSON
 - Object Detection Image / JSON
 -  Image Classification
 - License Plate Detection Image / JSON

# Motivation
We wanted to provide a an easy to use toolkit for Computer Vision tasks.

# Framework
We used Django for this project.


Django is a high-level Python framework. It is free and open-source, written in Python itself, and follows the model-view-template architectural pattern. We can use it to develop quality web applications faster and easier. Since developing for the web needs a set of similar components, you can use a framework. This way, you don’t have to reinvent the wheel. These tasks include authentication, forms, uploading files, management panels, and so.
# How to Run
1. Clone this repository.
2. Install the requirements using -
```sh
pip install -r requirements.txt
```

3. Run this repository from the the dltk-vision-core directory using- 
```sh
python manage.py runserver
```
4. Checkout the file for all possible to requests and formats to the project.


# Output

##  Face-Detection:
Face-Detection module uses HaarCascade algorigthm to detect and save the location of the detected face. Depending on the request used, it will either server those co-ordinates in json format or in base64 encoded image. 
### JSON
#### Request:

> http://127.0.0.1:8000/dltk-vision/face-detection/json

#### Response:
>
>{
>    "faces": [
>        {
>            "y": 102,
>            "x": 653,
>            "height": 39,
>            "width": 39
>        }
>}

The reponse gives the positions of faces across the image embedded in the POST Request

### Image
#### Request:

> http://127.0.0.1:8000/dltk-vision/face-detection/image

#### Response:
Insert Image here

The reponse gives the positions of faces across the image embedded in the POST Request

## Licence Plate Detection
We are detecting License plate using haarcascade Classifier trained on automobile license plates. Depending on the request used we can extract the JSON which contains the co-ordinates of the license plate or the image containing the license encoded in base64 string.
### Image
#### Request:

> http://127.0.0.1:8000/dltk-vision/licence-plate/json

#### Response:
insert image here

The reponse gives the co-ordinates of the license plates present  across the image embedded in the POST Request.

### JSON
#### Request:

> http://127.0.0.1:8000/dltk-vision/licence-plate/json

#### Response:
(sample output)
>{'y': 146, 'x': 132, 'height': 29, 'width': 114
>}

The reponse gives the co-ordinates of the license plates present  across the image embedded in the POST Request.

## Image Classification
We are using pre-trained InceptionV3 Model.Inception-v3 is a convolutional neural network that is 48 layers deep. You can load a pretrained version of the network trained on more than a million images from the ImageNet database. The pretrained network can classify images into 1000 object categories, such as keyboard, mouse, pencil, and many animals. As a result, the network has learned rich feature representations for a wide range of images. The network has an image input size of 299-by-299.
#### Request:

> http://127.0.0.1:8000/dltk-vision/image-classification

#### Response:
(sample output)
>{
>    "unicycle": "20.3%",
>    "jersey": "13.5%",
>    "knee_pad": "5.4%"
>}

The reponse gives the objects present  across the image embedded in the POST Request.

## Object Detection

### Image
#### Request:

> http://127.0.0.1:8000/dltk-vision/licence-plate/json

#### Response:
insert image here

The reponse gives the co-ordinates of the license plates present  across the image embedded in the POST Request.

### JSON
#### Request:

> http://127.0.0.1:8000/dltk-vision/licence-plate/json

#### Response:
(sample output)
>{'y': 146, 'x': 132, 'height': 29, 'width': 114
>}

The reponse gives the co-ordinates of the license plates present  across the image embedded in the POST Request.
