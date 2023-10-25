from app import celery
from app.main.stratus_py import stratus_s3_client
from flask import Response
import requests

@celery.task()
def download_object(filename, bucketname, access_key, secret_key):
    s3_client = stratus_s3_client(access_key, secret_key)
    file = filename.split('/')
    file_name = file[(len(file)-1)]
    file =  s3_client.get_object(Bucket = bucketname, Key = filename)
    return Response(
        file['Body'].read(),
        headers={"Content-Disposition": "attachment;filename=" + file_name}
    )