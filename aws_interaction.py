import os
import boto3  # pip install boto3

# Let's use Amazon S3
s3 = boto3.resource("s3")

# Print out bucket names
for bucket in s3.buckets.all():
    print(bucket.name)

# Create an s3 Bucket
s3 = boto3.client("s3")

s3.upload_file(
    Filename="C:\\Users\\robas\\Documents\\Code\\USD_Capstone\\data\\test_image.jpg",
    Bucket="salamon-testbucket",
    Key="test_image.jpg"
)

s3.download_file(
    Bucket="salamon-testbucket",
    Key="test_image.jpg",
    Filename="C:\\Users\\robas\\Documents\\Code\\USD_Capstone\\data\\test_image1.jpg"
)