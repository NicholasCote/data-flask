# os is used to get local environment variables 
import os
from flask import Response
# boto3 is the python package used to interact with S3
import boto3
import botocore
from botocore.exceptions import ClientError
from pathlib import Path
# This requests package is imported to disable certificate access warnings. 
# SSL certificates can be provided and this would not be required.
import requests.packages.urllib3
# We aren't verifying certs to start so this line is disable warnings
requests.packages.urllib3.disable_warnings()

# Define the Stratus S3 client to be used in other operations
def stratus_s3_client(endpoint, access_key, secret_key):
    # Create a boto3 sessions
    session = boto3.session.Session()
    # Create the S3 client based on the variables we set and provided
    s3_client = session.client(
        service_name='s3', 
        endpoint_url=endpoint, 
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        verify=False)
    # Return the client so that it can be used in other functions
    return s3_client

# Define the Stratus S3 resource to be used in other operations    
def stratus_s3_resource(endpoint, access_key, secret_key):
    # Create a boto3 sessions
    session = boto3.session.Session()
    # Create the S3 resource based on the variables we set and provided
    s3_resource = session.resource(
        service_name='s3', 
        endpoint_url=endpoint, 
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        verify=False)
    # Return the client so that it can be used in other functions
    return s3_resource

# Define a function to create a new S3 bucket with a name set by the bucket_name argument
# Requires an access_key with administrator privileges
def create_bucket(bucket_name, access_key, secret_key):
    # Use the S3 client already defined to make the call
    s3_client = stratus_s3_client(access_key, secret_key)
    # Call the create_bucket endpoint and provide the bucket_name specified by the user
    s3_client.create_bucket(Bucket=bucket_name)

# Define a function to list all buckets in the space
def list_all_buckets(endpoint, access_key, secret_key):
    buckets = []
    # Use the S3 client already defined to make the call
    s3_client = stratus_s3_client(endpoint, access_key, secret_key)
    # Get a response from the list_buckets endpoint
    response = s3_client.list_buckets()
    # Iterate through the Buckets in the response to print all the bucket names
    for bucket in response['Buckets']:
        # print(bucket['Name'])
        buckets.append(bucket['Name'])
    return buckets

# Define a function to list all the objects stored in a bucket
def list_bucket_objs(endpoint, bucket, access_key, secret_key):
    bucket_objs = []
    # Use the S3 resource already defined to make the call
    s3_resource = stratus_s3_resource(endpoint, access_key, secret_key)
    # Get the individual bucket resources for the bucket name provided in the function 
    bucket = s3_resource.Bucket(bucket)
    # Iterate through the response to show all objects contained within the bucket
    for obj in bucket.objects.all():
        #print(obj.key)
        bucket_objs.append(obj.key)
    return bucket_objs

# Define a function to upload a file/object to a bucket, specify the filename to upload and the bucket name to be placed in
def upload_file(endpoint, filename, bucketname, object_name, access_key, secret_key):
    if object_name is None:
            object_name = filename
    # Use the S3 client already defined to make the call
    s3_client = stratus_s3_client(endpoint, access_key, secret_key)
    # Use the upload_file endpoint to upload our filename to the specified bucket and keep the filename the same
    try:
        response = s3_client.upload_file(filename, bucketname, Key=object_name)
    except ClientError as e:
        print(e)
        return False
    print(filename + ' uploaded!')
    return True
    

# Define a function to download a file/object to a bucket
def download_fileobj(endpoint, filename, bucketname, access_key, secret_key):
    # Use the S3 client already defined to make the call
    s3_client = stratus_s3_client(endpoint, access_key, secret_key)
    # Open a local file with the same filename as the one we are downloading
    try:
        with open(filename, 'wb') as data:
            # Write the file to our open local file which is the python variable 'data'
            s3_client.download_fileobj(bucketname, filename, data)
    except FileNotFoundError:
        file_path = Path(filename)
        file_path.parent.mkdir(exist_ok=True, parents=True)
        with open(filename, 'wb') as data:
            s3_client.download_fileobj(bucketname, filename, data)

# Define a function to download a file/object to a bucket
def get_object(endpoint, filename, bucketname, access_key, secret_key):
    s3_client = stratus_s3_client(endpoint, access_key, secret_key)
    return s3_client.get_object(Bucket = bucketname, Key = filename)

# Define a function to download a file/object to a bucket
def user_download_fileobj(endpoint, filename, bucketname, access_key, secret_key):
    # Use the S3 client already defined to make the call
    # Open a local file with the same filename as the one we are downloading
    file = filename.split('/')
    file_name = file[(len(file)-1)]
    file = get_object(endpoint, filename, bucketname, access_key, secret_key)
    return Response(
        file['Body'].read(),
        headers={"Content-Disposition": "attachment;filename=" + file_name}
    )

        # Define a function to download a file/object to a bucket
def download_file(endpoint, filename, bucketname, access_key, secret_key):
    # Use the S3 client already defined to make the call
    s3_client = stratus_s3_client(endpoint, access_key, secret_key)
    if "/" in filename:
        directory_split = filename.split("/")
        if os.path.exists(directory_split[0]):
            # Open a local file with the same filename as the one we are downloading
            s3_client.download_file(bucketname, filename, filename)
        else:
            os.mkdir(directory_split[0])
            s3_client.download_file(bucketname, filename, filename)
    else:
            s3_client.download_file(bucketname, filename, filename)

def delete_object(endpoint, filename, bucketname, access_key, secret_key):
    s3_client = stratus_s3_client(endpoint, access_key, secret_key)
    del_response = s3_client.delete_object(Bucket = bucketname, Key = filename)
    if del_response['ResponseMetadata']['HTTPStatusCode'] == 204:
        message = 'Object deleted successfully'
    else:
        message = del_response['ResponseMetadata']['HTTPStatusCode'] + ' was returned instead of 204. Please check the bucket again for object status.'
    return message