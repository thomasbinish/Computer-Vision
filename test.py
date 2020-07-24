import cv2
import base64
import os
import time
from PIL import Image
import cv2
import imutils
import numpy as np
def licence_plate_image(image_path):
    print(os.getcwd())
    lp_cascade =cv2.CascadeClassifier("haarcascades/haarcascade_russian_plate_number.xml")
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    licence_plates = lp_cascade.detectMultiScale(gray, 5, 20)
    print(lp_cascade)
    for (x, y, w, h) in licence_plates:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
    cv2.imwrite('name.jpg',img)
    retval, buffer = cv2.imencode('.jpg', img)
    encoded_string = base64.b64encode(buffer)

    return encoded_string

def licence_plate_image_contour(image_path):

licence_plate_image_contour('license.jpg')
