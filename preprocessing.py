from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
import os
from google.cloud import storage

# Define Google Cloud Storage (GCS) bucket and dataset paths
GCS_BUCKET = "super-resolution-images"
HR_FOLDER = "DIV2K_train_HR"
LR_FOLDER = "DIV2K_train_LR"
HR_IMAGE_PATH = f"gs://{GCS_BUCKET}/{HR_FOLDER}/"
LR_IMAGE_PATH = f"gs://{GCS_BUCKET}/{LR_FOLDER}/"

spark = SparkSession.builder \
    .appName("DIV2K Preprocessing") \
    .config("spark.hadoop.fs.gs.impl", "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem") \
    .config("spark.hadoop.google.cloud.auth.service.account.enable", "true") \
    .getOrCreate()

# Initialize GCS Client
storage_client = storage.Client()

def read_image_from_gcs(image_path):
    """Reads an image from Google Cloud Storage and converts it to a NumPy array."""
    bucket_name = image_path.split('/')[2]
    blob_path = '/'.join(image_path.split('/')[3:])
    
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    image_bytes = blob.download_as_bytes()
    
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    return np.array(image)

def generate_lr_image(image_path):
    """Reads an HR image from GCS, downsamples it (x4), and saves it as an LR image in GCS."""

    # Initialize GCS Client inside the function
    storage_client = storage.Client()

    # Extract bucket name and file path
    bucket_name = image_path.split('/')[2]
    blob_path = '/'.join(image_path.split('/')[3:])

    # Read image from GCS
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    image_bytes = blob.download_as_bytes()

    # Convert image to NumPy array
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    hr_image = np.array(image)

    # Downscale using bicubic interpolation (x4)
    lr_image = cv2.resize(hr_image, (hr_image.shape[1] // 4, hr_image.shape[0] // 4), interpolation=cv2.INTER_CUBIC)

    # Generate the correct LR image path
    filename = os.path.basename(image_path)  # Get only the filename
    lr_image_path = f"{LR_IMAGE_PATH}{filename}"  # Ensure correct storage path in GCS

    # Upload LR image to GCS
    bucket = storage_client.bucket(GCS_BUCKET)
    blob = bucket.blob(f"{LR_FOLDER}/{filename}")

    _, buffer = cv2.imencode('.png', lr_image)
    blob.upload_from_string(buffer.tobytes(), content_type='image/png')

    return lr_image_path


# Register UDF for Spark processing
generate_lr_udf = udf(generate_lr_image, StringType())

# Load HR image paths from GCS into Spark DataFrame
hr_image_paths = [
    f"gs://{GCS_BUCKET}/{HR_FOLDER}/{blob.name.split('/')[-1]}"
    for blob in storage_client.bucket(GCS_BUCKET).list_blobs(prefix=HR_FOLDER)
]

hr_image_df = spark.createDataFrame([(path,) for path in hr_image_paths], ["image_path"])

# Generate and store LR images using Spark
lr_image_df = hr_image_df.withColumn("lr_image_path", generate_lr_udf(col("image_path")))

# Show results
lr_image_df.show()

# Stop Spark session
spark.stop()
