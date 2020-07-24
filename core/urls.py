from django.conf.urls import url

from core.controller.facial_detection_controller import *
from core.controller.job_controller import job_status, job_list
from core.controller.licence_plate_detection_controller import *
from core.controller.image_classification_controller import *
from core.controller.object_detection_controller import *

urlpatterns = [
        url(r'^object-detection/json$', obj_json, name='object_detection_json'),
        url(r'^object-detection/image', obj_image, name='object_detection_image'),

        url(r'^custom-object-detection/train$', custom_obj_detection_train_controller, name='custom_obj_detection_train'),
        url(r'^custom-object-detection/predict$', custom_obj_detection_predict_controller, name='custom_obj_detection_predict'),
        url(r'^custom-object-detection/predict/image$', custom_obj_detection_predict_image_controller, name='custom_obj_detection_predict'),
        url(r'^custom-object-detection/predict/json$', custom_obj_detection_predict_json_controller, name='custom_obj_detection_predict'),

        url(r'^face-detection/image$', face_detection_image_controller, name='face_detection_image'),
        url(r'^face-detection/json$', face_detection_json_controller, name='face_detection_json'),

        url(r'^licence-plate/image$', license_plate_image_controller, name='licence-plate_image'),
        url(r'^licence-plate/json$', license_plate_json_controller, name='licence-plate_json'),

        url(r'^custom-image-classification/train$', custom_image_clf_train_controller, name='custom_image_clf_train'),
        url(r'^custom-image-classification/predict$', custom_image_clf_predict_controller, name='custom_image_clf_predict'),
        url(r'^custom-image-classification/predict/image$', custom_image_clf_predict_image_controller, name='custom_image_clf_predict_image'),
        url(r'^image-classification$', image_clf_controller, name='image_clf'),

        url(r'^status/(?P<job_id>.+)$', job_status, name='job_status'),
        url(r'^list', job_list, name='job_list'),
]
