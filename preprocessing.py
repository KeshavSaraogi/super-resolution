from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType
import boto3
import io
from PIL import Image

# Constants
S3_BUCKET = "images-resolution"
HR_FOLDER = "DIV2K_train_HR"
LR_FOLDER = "DIV2K_train_LR"

# Initialize Spark session
spark = SparkSession.builder.appName("DIV2K Preprocessing").getOrCreate()

# Initialize S3 client (ONCE, outside UDF)
s3_client = boto3.client("s3")

def process_image(hr_image_path):
    """
    Download HR image from S3, generate LR image in-memory, and upload it back to S3.
    """
    try:
        # Extract bucket name and key
        bucket_name, key = hr_image_path.replace("s3://", "").split("/", 1)

        # Read image from S3
        response = s3_client.get_object(Bucket=bucket_name, Key=key)
        img = Image.open(io.BytesIO(response["Body"].read())).convert("RGB")

        # Resize image (bicubic downsampling)
        lr_img = img.resize((img.width // 4, img.height // 4), Image.BICUBIC)

        # Convert image to bytes for direct S3 upload
        buffer = io.BytesIO()
        lr_img.save(buffer, format="PNG")
        buffer.seek(0)

        # Generate LR image S3 path
        lr_key = key.replace(HR_FOLDER, LR_FOLDER)

        # Upload LR image to S3
        s3_client.put_object(Bucket=S3_BUCKET, Key=lr_key, Body=buffer, ContentType="image/png")

        return f"s3://{S3_BUCKET}/{lr_key}"
    except Exception as e:
        print(f"Error processing {hr_image_path}: {e}")
        return None

# Fetch HR image paths from S3 (OUTSIDE UDF)
hr_objects = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=HR_FOLDER).get("Contents", [])
hr_image_paths = [f"s3://{S3_BUCKET}/{obj['Key']}" for obj in hr_objects if obj["Key"].endswith(".png")]

# Process images in a loop (NO UDF)
processed_images = [(path, process_image(path)) for path in hr_image_paths]

# Convert to DataFrame
df = spark.createDataFrame(processed_images, ["image_path", "lr_image_path"])

# Show results
df.show()

# Stop Spark session
spark.stop()
