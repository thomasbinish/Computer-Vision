import json
import traceback
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt

from core.component.request_component import pretty_request
from core.constants import CONTENT_TYPE
from core.lib.license_plate_detection import *


@csrf_exempt
def license_plate_image_controller(request):
    try:
        if request.method == "POST":
            print(pretty_request(request))
            with open('lp.png', 'wb') as f:
                f.write(request.FILES['image'].read())
            image_path = "lp.png"
            response = licence_plate_image(image_path)
            return HttpResponse(response, content_type=CONTENT_TYPE, status=200)
    except Exception as e:
        traceback.print_exc()
        return HttpResponse(e.args, status=500, content_type=CONTENT_TYPE)


@csrf_exempt
def license_plate_json_controller(request):
    try:
        if request.method == "POST":
            print(pretty_request(request))
            with open('original_image', 'wb') as f:
                f.write(request.FILES['image'].read())
            image_path = "original_image"
            response = licence_plate_json(image_path)
            return HttpResponse(json.dumps(response), content_type=CONTENT_TYPE, status=200)
    except Exception as e:
        traceback.print_exc()
        return HttpResponse(e.args, status=500, content_type=CONTENT_TYPE)
