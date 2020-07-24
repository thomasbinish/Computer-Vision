from core.utils.eureka_utils import get_service_url
from properties import *

CHARSET = 'utf-8'
CONTENT_TYPE = "application/json"
if REGISTRY:
    STORAGE_URL = get_service_url(STORAGE_SERVICE)

# task status
RUN = "RUN"
FINISH = "FINISH"
FAIL = "FAIL"

# Tasks
TRAIN = "TRAIN"
PREDICT = "PREDICT"
