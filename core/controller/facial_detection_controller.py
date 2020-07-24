import json
import traceback
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt

from core.component.request_component import pretty_request
from core.constants import CONTENT_TYPE
from core.lib.face_detection import *


@csrf_exempt
def face_detection_image_controller(request):
    try:
        if request.method == "POST":
            pretty_request(request)
            with open('original_image', 'wb') as f:
                f.write(request.FILES['image'].read())
            image_path = "original_image"
            response = face_detection_image(image_path)
            return HttpResponse(response, content_type=CONTENT_TYPE, status=200)
    except Exception as e:
        traceback.print_exc()
        return HttpResponse(e.args, status=500, content_type=CONTENT_TYPE)


@csrf_exempt
def face_detection_json_controller(request):
    try:
        if request.method == "POST":
            print(pretty_request(request))
            with open('original_image', 'wb') as f:
                f.write(request.FILES['image'].read())
            image_path = "original_image"
            response = face_detection_json(image_path)
            output = {
                "faces":response
            }
            return HttpResponse(json.dumps(output), content_type=CONTENT_TYPE, status=200)
    except Exception as e:
        traceback.print_exc()
        return HttpResponse(e.args, status=500, content_type=CONTENT_TYPE)


@csrf_exempt
def eyes_detection(request):
    try:
        if request.method == "POST":
            print(pretty_request(request))
            with open('original_image', 'wb') as f:
                f.write(request.FILES['image'].read())
            image_path = "original_image"
            response = _(image_path)
            return HttpResponse(json.dumps(response), content_type=CONTENT_TYPE, status=200)
    except Exception as e:
        traceback.print_exc()
        return HttpResponse(e.args, status=500, content_type=CONTENT_TYPE)


@csrf_exempt
def smile_detection(request):
    try:
        if request.method == "POST":
            print(pretty_request(request))
            with open('original_image', 'wb') as f:
                f.write(request.FILES['image'].read())
            image_path = "original_image"
            response = _(image_path)
            return HttpResponse(response, content_type=CONTENT_TYPE, status=200)
    except Exception as e:
        traceback.print_exc()
        return HttpResponse(e.args, status=500, content_type=CONTENT_TYPE)
