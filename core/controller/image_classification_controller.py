import json
import traceback
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from core.component.custom_img_clf_component import train_component, predict_component
from core.component.job_component import create_job
from core.component.request_component import pretty_request
from core.constants import *
from core.lib.custom_image_classification import prediction_image
from core.lib.image_classification import *
from core.po.pred_request_image import ImagePredictionRequest
from core.po.prediction_request import PredictionRequest
from core.po.image_clf_request import ImageClfRequest


@csrf_exempt
def custom_image_clf_train_controller(request):
    if request.method == "POST":
        pretty_request(request)
        params = json.loads(request.body.decode(CHARSET))
        user_id = request.META['HTTP_USER_ID']
        train_request = ImageClfRequest(params)
        job_id = create_job(user_id, TRAIN, train_request.service, train_request.name)
        train_component.delay(params, user_id, job_id)
        response = {"job_id":job_id}
        return HttpResponse(json.dumps(response), content_type=CONTENT_TYPE, status=200)


@csrf_exempt
def custom_image_clf_predict_controller(request):
    if request.method == "POST":
        print(pretty_request(request))
        params = json.loads(request.body.decode(CHARSET))
        user_id = request.META['HTTP_USER_ID']
        prediction_request = PredictionRequest(params)
        job_id = create_job(user_id, PREDICT, prediction_request.service, prediction_request.name)
        predict_component.delay(params, user_id, job_id)
        response = {"job_id": job_id}
        return HttpResponse(json.dumps(response), content_type=CONTENT_TYPE, status=200)


@csrf_exempt
def custom_image_clf_predict_image_controller(request):
    try:
        if request.method == "POST":
            print(pretty_request(request))
            with open('clf_image.jpg', 'wb') as f:
                f.write(request.FILES['image'].read())
            image_path = "clf_image.jpg"
            model_url = request.POST["modelUrl"]
            user_id = request.META['HTTP_USER_ID']
            response = prediction_image(image_path, model_url, user_id)
            return HttpResponse(json.dumps(response), content_type=CONTENT_TYPE, status=200)
    except Exception as e:
        traceback.print_exc()
        return HttpResponse(e.args, status=500, content_type=CONTENT_TYPE)


@csrf_exempt
def image_clf_controller(request):
    try:
        if request.method == "POST":
            print(pretty_request(request))
            with open('clf_image.jpg', 'wb') as f:
                f.write(request.FILES['image'].read())
            image_path = "clf_image.jpg"
            response = image_classification(image_path)
            return HttpResponse(json.dumps(response), content_type=CONTENT_TYPE, status=200)
    except Exception as e:
        traceback.print_exc()
        return HttpResponse(e.args, status=500, content_type=CONTENT_TYPE)
