import traceback
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt

from core.component.custom_obj_detection_component import objd_train_component, objd_predict_component
from core.component.job_component import create_job
from core.component.request_component import pretty_request
from core.constants import *
from core.lib.object_detection import object_detection_image, object_detection_json
from core.lib.custom_object_detection import *
from core.po.object_detection_request import ObjDetectionRequest
from core.po.prediction_request import PredictionRequest


@csrf_exempt
def obj_image(request):
    try:
        if request.method == "POST":
            print(pretty_request(request))
            with open('original_image.png', 'wb') as f:
                f.write(request.FILES['image'].read())
            image_path = "original_image.png"
            response = object_detection_image(image_path)
            return HttpResponse(response, content_type=CONTENT_TYPE, status=200)
    except Exception as e:
        traceback.print_exc()
        return HttpResponse(e.args, status=500, content_type=CONTENT_TYPE)


@csrf_exempt
def obj_json(request):
    try:
        if request.method == "POST":
            print(pretty_request(request))
            with open('original_image.png', 'wb') as f:
                f.write(request.FILES['image'].read())
            image_path = "original_image.png"
            response = object_detection_json(image_path)
            return HttpResponse(json.dumps(response), content_type=CONTENT_TYPE, status=200)
    except Exception as e:
        traceback.print_exc()
        return HttpResponse(e.args, status=500, content_type=CONTENT_TYPE)


@csrf_exempt
def custom_obj_detection_train_controller(request):
    if request.method == "POST":
        print(pretty_request(request))
        params = json.loads(request.body.decode(CHARSET))
        user_id = request.META['HTTP_USER_ID']
        train_request = ObjDetectionRequest(params)
        job_id = create_job(user_id, TRAIN, train_request.service, train_request.name)
        objd_train_component.delay(params, user_id, job_id)
        response = {"job_id": job_id}
        return HttpResponse(json.dumps(response), content_type=CONTENT_TYPE, status=200)


@csrf_exempt
def custom_obj_detection_predict_controller(request):
    if request.method == "POST":
        print(pretty_request(request))
        params = json.loads(request.body.decode(CHARSET))
        user_id = request.META['HTTP_USER_ID']
        prediction_request = PredictionRequest(params)
        job_id = create_job(user_id, PREDICT, prediction_request.service, prediction_request.name)
        objd_predict_component.delay(params, user_id, job_id)
        response = {"job_id": job_id}
        return HttpResponse(json.dumps(response), content_type=CONTENT_TYPE, status=200)


@csrf_exempt
def custom_obj_detection_predict_image_controller(request):
    try:
        if request.method == "POST":
            print(pretty_request(request))
            with open('original_image.png', 'wb') as f:
                f.write(request.FILES['image'].read())
            image_path = "original_image.png"
            model_url = request.POST["modelUrl"]
            user_id = request.META['HTTP_USER_ID']
            response = custom_object_detection_image(model_url, image_path, user_id)
            return HttpResponse(response, content_type=CONTENT_TYPE, status=200)
    except Exception as e:
        traceback.print_exc()
        return HttpResponse(e.args, status=500, content_type=CONTENT_TYPE)


@csrf_exempt
def custom_obj_detection_predict_json_controller(request):
    try:
        if request.method == "POST":
            print(pretty_request(request))
            with open('original_image.png', 'wb') as f:
                f.write(request.FILES['image'].read())
            image_path = "original_image.png"
            model_url = request.POST["modelUrl"]
            user_id = request.META['HTTP_USER_ID']
            response = custom_object_detection_json(model_url, image_path, user_id)
            return HttpResponse(json.dumps(response), content_type=CONTENT_TYPE, status=200)
    except Exception as e:
        traceback.print_exc()
        return HttpResponse(e.args, status=500, content_type=CONTENT_TYPE)
