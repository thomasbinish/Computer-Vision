# Computer Vision
## Description
DLTK Computer Vision enables you to find meaning in visual content! Analyze images for scenes, objects, faces, and other content. Choose a default model off the shelf, or create your own custom classifier. Develop smart applications that analyze the visual content of images or video frames to understand what is happening in a scene.

## Features provided
DLTK Computer Vision Module provides the following APIs as of now:
1. **Face Detection Image/ JSON**: Uses HaarCascade algorigthm to detect and save the location of the detected face. Depending on the request used, it will either server those co-ordinates in json format or in base64 encoded image. 
2. **Object Detection Image/ JSON**: Detect multiple objects in the same image using RetinaNet-50. It also tags the objects and shows their location within the image. 
3. **Image Classification**: Used pre-trained InceptionV3 Model which is a convolutional neural network having 48 hidden layers. The pretrained network can classify images into 1000 object categories, such as keyboard, mouse, pencil, and many animals. The network has an image input size of 299-by-299.
4. **License Plate Detection Image/ JSON**: Used Haarcascade Classifier trained on automobile license plates. Depending on the request used we can extract the JSON which contains the co-ordinates of the license plate or the image containing the license encoded in base64 string.

## Demo

**Face Detection**
![face detection](https://github.com/dltk-ai/Computer-Vision/blob/master/CVimages/group.jpg)
![face detection](https://github.com/dltk-ai/Computer-Vision/blob/master/CVimages/face_detect.jpeg)

**Object Detection**
![Object Detection](https://github.com/dltk-ai/Computer-Vision/blob/master/CVimages/japan.jpg)
![Object Detection](https://github.com/dltk-ai/Computer-Vision/blob/master/CVimages/image%20(1).jpg)

**License plate detection**

![License plate detection](https://github.com/dltk-ai/Computer-Vision/blob/master/CVimages/license.jpeg)
![License plate detection](https://github.com/dltk-ai/Computer-Vision/blob/master/CVimages/out.jpg)


# Motivation
This Repository is created to show how DLTK computer vision API uses advanced deep learning algorithms to analyze images and videos for scenes, objects, faces, licence plates and other content. For example, you upload a photograph and service detects different objects in a photograph. You can use the default model from DLTK.AI or create your own custom classifier.

# Frameworks/ Tech Stack used
1. [Django](https://www.djangoproject.com/) : Python-based open-source web framework that follows the model-view-template (MVT) architectural pattern.
2. [OpenCV](https://opencv.org/): Library of programming functions mainly aimed at real-time computer vision.
3. [InceptionV3](https://keras.io/api/applications/inceptionv3/): convolutional neural network having 48 hidden layers. The pretrained network can classify images into 1000 object categories, such as keyboard, mouse, pencil, and many animals.
4. [RetinaNet-50](https://keras.io/examples/vision/retinanet/): a popular single-stage detector, which is accurate and runs fast. RetinaNet uses a feature pyramid network to efficiently detect objects at multiple scales and introduces a new loss, the Focal loss function, to alleviate the problem of the extreme foreground-background class imbalance.
5. [Haar Cascade](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_objdetect/py_face_detection/py_face_detection.html): Machine learning based approach where a cascade function is trained from a lot of positive and negative images. It is then used to detect objects in other images.

## How to use?

Before executing this project, first, we need to download the models used for Computer Vision tasks in the existing repository. Due to the sheer size of the models, we decided not to use GitHub and store in Amazon S3 buckets.

To download the models-
`wget https://dltk-ai-prod.s3.ap-south-1.amazonaws.com/computer_vision_models/resources.zip`

Then the 'resources.zip' needs to unzipped inside the Computer-Vision repository. The unzipped directory 'resources' contains all models that we use in this repository.

**Option-1**: Executing ***dltk-vision-core*** as a service. 

1. Clone the repository
2. Install all the required dependencies.
`pip install requirements.txt` 
3. Open command prompt/Terminal and run the django server 
`python manage.py runserver 0.0.0.0:8187`
4. Start using the APIs listed below:

**Face detection API:**
`curl --location --request POST 'http://0.0.0.0:8187/dltk-vision/face-detection/image' \
--form 'image=@image_path'`

JSON:

`curl --location --request POST 'http://0.0.0.0:8187/dltk-vision/face-detection/json' \
--form 'image=@image_path'`


**Object detection API:**
`curl --location --request POST 'http://0.0.0.0:8187/dltk-vision/object-detection/image' \
--form 'image=@image_path'`

JSON:

`curl --location --request POST 'http://0.0.0.0:8187/dltk-vision/object-detection/json' \
--form 'image=@image_path'`


**Image classification API:**
`curl --location --request POST 'http://0.0.0.0:8187/dltk-vision/image-classification' \
--form 'image=@image_path'`



**Option-2**: Executing ***dltk-vision-core*** as a docker container.

**Docker**: Docker is an advanced OS virtualization software platform that makes it easier to create, deploy, and run applications in a Docker container.

Install Docker by following this [link](https://docs.docker.com/get-docker/).

**Docker compose**: Docker Compose is that users can activate all the services (containers) using a single command.

Install Docker Compose by following this [link](https://docs.docker.com/compose/install/)

Steps:

1. Clone the repository;
2. Go to the path where docker-compose.yml is placed.
3. Run the command to start the container `sudo docker-compose up -d`
4. Now check the containers `sudo docker ps`
![docker-output](https://github.com/dltk-ai/Computer-Vision/blob/master/CVimages/docker-cv.png)
5. Execute the CURL Command mentioned in option-1
6. Run the command to stop the container `sudo docker-compose down`

## Founding Members
[![](https://github.com/shreeramiyer.png?size=50)](https://github.com/shreeramiyer)

## Lead Maintainer
[![](https://github.com/GHub4Naveen.png?size=50)](https://github.com/GHub4Naveen)
## Core Mainteiners
[![](https://github.com/dltk-ai.png?size=50)](https://github.com/dltk-ai)
## Core Contributers 
[![](https://github.com/SivaramVeluri15.png?size=50)](https://github.com/SivaramVeluri15)
[![](https://github.com/vishnupeesapati.png?size=50)](https://github.com/vishnupeesapati)
[![](https://github.com/EpuriHarika.png?size=50)](https://github.com/EpuriHarika/)
[![](https://github.com/nageshsinghc4.png?size=50)](https://github.com/nageshsinghc4)
[![](https://github.com/appareddyraja.png?size=50)](https://github.com/appareddyraja)
[![](https://github.com/shakeeldhada.png?size=50)](https://github.com/shakeeldhada)


## License
