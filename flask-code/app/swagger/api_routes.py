from app import api, app
from flask_restful import Resource
from app.main.stratus_py import stratus_s3_client, list_all_buckets
from flask import Response, session
from flask import jsonify
import os

class HellowWorld(Resource):
    def get(self):
        return jsonify({'message':'Hello, World!'})
    
api.add_resource(HellowWorld, '/api/hello')

class S3Download(Resource):
    def get(self):
        s3_client = stratus_s3_client(access_key, secret_key)
        file = filename.split('/')
        file_name = file[(len(file)-1)]
        file =  s3_client.get_object(Bucket = bucketname, Key = filename)
        return Response(
            file['Body'].read(),
            headers={"Content-Disposition": "attachment;filename=" + file_name}
        )

api.add_resource(S3Download, '/api/S3Download')

class GLADE(Resource):
    def get(self, path):
        return os.listdir(path)

api.add_resource(GLADE, '/api/GLADE')

class ListBuckets(Resource):
    def post(self, access_id, secret_key):
        buckets = list_all_buckets(access_id, secret_key)
        return jsonify({'buckets':buckets})

api.add_resource(ListBuckets, '/api/ListBuckets/<access_id>:<secret_key>')