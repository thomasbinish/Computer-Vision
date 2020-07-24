import traceback

from billiard.process import current_process

from core import constants
from core.component.job_component import create_job, update_job
from core.lib.custom_object_detection import custom_obj_detection_train, custom_obj_detection_predict
from core.po.prediction_request import PredictionRequest
from core.po.object_detection_request import ObjDetectionRequest
from dltk_vision.celery import app


@app.task(bind=True)
def objd_train_component(self, params, user_id, job_id):
    train_request = ObjDetectionRequest(params)
    worker = current_process().index
    try:
        response = custom_obj_detection_train(train_request, user_id, worker)
        output = {
            "modelUrl": response["fileUrl"],
            "modelName": train_request.name,
            "loss":0,
            "job_id": job_id,
            "user_id": user_id,
        }
        update_job(job_id, output, constants.FINISH)
        return output
    except Exception as e:
        traceback.print_exc()
        output = {
            "modelName": train_request.name,
            "cause": e.args,
            "job_id": job_id,
            "user_id": user_id
        }
        update_job(job_id, output, constants.FAIL)
        return e.args


@app.task(bind=True)
def objd_predict_component(self, params, user_id, job_id):
    prediction_request = PredictionRequest(params)
    worker = current_process().index
    try:
        response = custom_obj_detection_predict(prediction_request, user_id, worker)
        output = {
            "predUrl": response["fileUrl"],
            "job_id": job_id,
            "user_id": user_id,
        }
        update_job(job_id, output, constants.FINISH)
        return output
    except Exception as e:
        traceback.print_exc()
        output = {
            "modelName": prediction_request.name,
            "cause":e.args,
            "job_id":job_id,
            "user_id":user_id
        }
        update_job(job_id, output, constants.FAIL)
        return e.args
