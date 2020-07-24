import base64
import os
import cv2


def face_detection_image(image_path):
    try:
        face_cascade = cv2.CascadeClassifier('resources/haarcascade_frontalface_default.xml')
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 3)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 1)
        cv2.imwrite('img.jpg', img)
        with open("img.jpg", "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
    finally:
        os.remove("img.jpg")
        os.remove(image_path)
    return encoded_string


def face_detection_json(image_path):
    try:
        face_cascade = cv2.CascadeClassifier('resources/haarcascade_frontalface_default.xml')
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 3)
        response = []
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 1)
            face = {"x": int(x), "y": int(y), "width": int(w), "height": int(h)}
            response.append(face)
        return response
    finally:
        os.remove(image_path)
