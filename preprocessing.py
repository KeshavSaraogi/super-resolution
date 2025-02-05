from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
import os
import boto3

# Define AWS S3 bucket and dataset paths
S3_BUCKET = "images-resolution"
HR_FOLDER = "DIV2K_train_HR"
LR_FOLDER = "DIV2K_train_LR_bicubic_X4"
HR_IMAGE_PATH = f"s3://{S3_BUCKET}/{HR_FOLDER}/"
LR_IMAGE_PATH = f"s3://{S3_BUCKET}/{LR_FOLDER}/"

spark = SparkSession.builder \
    .appName("DIV2K Preprocessing") \
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
    .getOrCreate()

# Initialize S3 Client
s3_client = boto3.client("s3")

def read_image_from_s3(image_path):
    """Reads an image from S3 and converts it to a NumPy array."""
    key = "/".join(image_path.split("/")[3:])
    response = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
    image_bytes = response["Body"].read()
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    return np.array(image)

def generate_lr_image(image_path):
    """Reads an HR image from S3, downsamples it (x4), and saves it as an LR image in S3."""
    key = "/".join(image_path.split("/")[3:])
    response = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
    image_bytes = response["Body"].read()
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    hr_image = np.array(image)

    # Downscale using bicubic interpolation (x4)
    lr_image = cv2.resize(hr_image, (hr_image.shape[1] // 4, hr_image.shape[0] // 4), interpolation=cv2.INTER_CUBIC)

    # Generate LR image path
    filename = os.path.basename(image_path)
    lr_image_key = f"{LR_FOLDER}/{filename}"

    # Upload LR image to S3
    _, buffer = cv2.imencode('.png', lr_image)
    s3_client.put_object(Bucket=S3_BUCKET, Key=lr_image_key, Body=buffer.tobytes(), ContentType='image/png')

    return f"s3://{S3_BUCKET}/{lr_image_key}"

# Register UDF for Spark processing
generate_lr_udf = udf(generate_lr_image, StringType())

# Load HR image paths from S3 into Spark DataFrame
hr_image_paths = [
    f"s3://{S3_BUCKET}/{HR_FOLDER}/{obj['Key'].split('/')[-1]}"
    for obj in s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=HR_FOLDER)["Contents"]
]

hr_image_df = spark.createDataFrame([(path,) for path in hr_image_paths], ["image_path"])

# Generate and store LR images using Spark
lr_image_df = hr_image_df.withColumn("lr_image_path", generate_lr_udf(col("image_path")))

# Show results
lr_image_df.show()

# Stop Spark session
spark.stop()
