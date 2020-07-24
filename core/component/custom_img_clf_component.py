import traceback

from core import constants
from core.component.job_component import create_job, update_job
from core.lib.custom_image_classification import custom_image_classification, prediction_zip
from core.po.prediction_request import PredictionRequest
from core.po.image_clf_request import ImageClfRequest
from dltk_vision.celery import app
from billiard.process import current_process


@app.task(bind=True)
def train_component(self, params, user_id, job_id):
    train_request = ImageClfRequest(params)
    worker = current_process().index
    try:
        model_file, loss, accuracy = custom_image_classification(train_request, user_id, worker)
        output = {
            "modelUrl": model_file["fileUrl"],
            "fileId": model_file["id"],
            "modelName":train_request.name,
            "accuracy": accuracy,
            "loss": loss,
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
def predict_component(self, params, user_id, job_id):
    prediction_request = PredictionRequest(params)
    worker = current_process().index
    try:
        response = prediction_zip(prediction_request, user_id, worker)
        output = {
            "predUrl": response["fileUrl"],
            "fileId": response["id"],
            "job_id": job_id,
            "user_id": user_id,
        }
        update_job(job_id, output, constants.FINISH)
        return output
    except Exception as e:
        traceback.print_exc()
        output = {
            "modelName": prediction_request.name,
            "cause": e.args,
            "job_id": job_id,
            "user_id": user_id
        }
        update_job(job_id, output, constants.FAIL)
        return e.args