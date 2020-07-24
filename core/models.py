from __future__ import unicode_literals
from django.db import models
from django.contrib.postgres.fields import JSONField


class JobStatus(models.Model):
    job_id = models.BigAutoField(default=None, primary_key=True)
    user_id = models.BigIntegerField(null=True)
    task = models.CharField(max_length=50, null=True)
    task_status = models.CharField(max_length=100, default="START")
    output = JSONField(default={})
    service = models.CharField(max_length=100, default=" ")

    def __unicode__(self):
        return str(self.job_id)

    def __str__(self, ):
        return str(self.job_id)

    def save(self, force_insert=False, force_update=False, using=None):
        super(JobStatus, self).save(force_insert=False, force_update=False, using=None)


class ModelInfo(models.Model):
    id = models.BigAutoField(default=None, primary_key=True)
    user_id = models.BigIntegerField(null=True)
    model_url_s3 = models.CharField(max_length=200, null=True)
    model_path = models.CharField(max_length=200,  null=True)

    def __unicode__(self):
        return str(self.id)

    def __str__(self, ):
        return str(self.id)

    def save(self, force_insert=False, force_update=False, using=None):
        super(ModelInfo, self).save(force_insert=False, force_update=False, using=None)