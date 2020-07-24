from core import constants
from core.models import JobStatus


def create_job(user_id, task, service, name):
    job = JobStatus()
    job.user_id = user_id
    job.service = service.lower()
    job.task_status = constants.RUN
    job.output = {"modelName": name}
    job.task = task
    job.save()
    return job.job_id


def update_job(job_id, output, state):
    job = JobStatus.objects.get(job_id=job_id)
    job.task_status = state
    job.output = output
    job.save()