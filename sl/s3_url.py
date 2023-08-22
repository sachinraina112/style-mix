import logging
import boto3
from botocore.exceptions import ClientError
import os
from configs.AppConfig import s3_config

from boto3.session import Session
from configs.Security import Security

def decrypt(var):
    sec = Security()
    byte_val = sec.string_to_bytes(var)
    value = sec.decrypt(byte_val)
    return value



def get_url(file, up=True):
    """Verify both boto3 and botocore clients stay in sync."""
    bucket = s3_config["bucket"]
    user = str(decrypt(s3_config["user"]))
    pwd = str(decrypt(s3_config["key"]))
    region = 'ap-south-1'
    client = Session(
            aws_access_key_id=user,
            aws_secret_access_key=pwd,
            region_name=region,
        ).client("s3")
    s3_path = file[2:]
    if up:
        client.upload_file(file,bucket, s3_path)
    else:
        client.download_file(file, bucket, s3_path)
    url = client.generate_presigned_url('get_object', Params = {'Bucket': 'stybucket', 'Key': s3_path}, ExpiresIn = 6000)
    return url


# def upload_file(file_name, bucket, object_name=None):
#     """Upload a file to an S3 bucket

#     :param file_name: File to upload
#     :param bucket: Bucket to upload to
#     :param object_name: S3 object name. If not specified then file_name is used
#     :return: True if file was uploaded, else False
#     """

#     # If S3 object_name was not specified, use file_name
#     if object_name is None:
#         object_name = os.path.basename(file_name)

#     # Upload the file
#     s3_client = boto3.client('s3')
#     try:
#         response = s3_client.upload_file(file_name, bucket, object_name)
#     except ClientError as e:
#         logging.error(e)
#         return False
#     return True






if __name__ == "__main__":
    # file_name = "Output_save1-2ca8ad2d-b540-45cc-99cf-f3f97b678ca0.jpg"
    # bucket = "stybucket"
    # print(upload_file(file_name, bucket, object_name=None))
    print(get_url())