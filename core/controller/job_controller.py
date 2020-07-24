import json
import traceback

from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt

from core import constants
from core.component.request_component import pretty_request
from core.constants import CONTENT_TYPE, CHARSET
from core.exceptions import JobNotFoundError
from core.models import JobStatus


@csrf_exempt
def job_status(request, job_id):
    try:
        if request.method == "GET":
            print(pretty_request(request))
            job = JobStatus.objects.get(job_id=job_id)
            response = {
                "task": job.task,
                "status" : job.task_status,
                "output": job.output
            }
            if response is not None:
                return HttpResponse(json.dumps(response), content_type=CONTENT_TYPE, status=200)
            else:
                raise JobNotFoundError()
    except Exception as e:
        traceback.print_exc()
        return HttpResponse(e.args, status=500, content_type=CONTENT_TYPE)


@csrf_exempt
def job_list(request):
    try:
        if request.method == "POST":
            print(pretty_request(request))
            user_id = request.META['HTTP_USER_ID']
            params = json.loads(request.body.decode(CHARSET))
            status  = params['status']
            service = params['service']
            jobs_list = []
            if status == "ALL":
                jobs_list = JobStatus.objects.filter(user_id=user_id, task=constants.TRAIN, service=service).values()
            elif status == constants.FINISH:
                jobs_list = JobStatus.objects.filter(user_id=user_id, task_status=constants.FINISH, task=constants.TRAIN, service=service).values()
            response = []
            for job in jobs_list:
                x = {
                    "status" : job["task_status"],
                    "output": job["output"]
                }
                response.append(x)
            output={}
            output["job_list"] = response
            return HttpResponse(json.dumps(output), content_type=CONTENT_TYPE, status=200)
    except Exception as e:
        traceback.print_exc()
        return HttpResponse(e.args, status=500, content_type=CONTENT_TYPE)