import base64
import os
from PIL import Image
import cv2
import imutils
import numpy as np




def licence_plate_image(image_path):
    lp_cascade =cv2.CascadeClassifier("haarcascades/indian_license_plate.xml")
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    licence_plates = lp_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    print(lp_cascade)
    for (x, y, w, h) in licence_plates:
        cv2.rectangle(img, (x, y), (x + w, y + h), (51, 255, 51), 5)
    retval, buffer = cv2.imencode('.jpg', img)
    encoded_string = base64.b64encode(buffer)

    return encoded_string


def licence_plate_json(image_path):
    try:
        lp_cascade = cv2.CascadeClassifier('haarcascades/indian_license_plate.xml')
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        licence_plates = lp_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        response = []
        for (x, y, w, h) in licence_plates:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            lp = {"x": int(x), "y": int(y), "width": int(w), "height": int(h) }
            response.append(lp)
        return response
    finally:
        os.remove(image_path)
