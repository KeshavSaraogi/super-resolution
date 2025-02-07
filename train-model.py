from pyspark.sql import SparkSession
import boto3
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import io

# Constants
S3_BUCKET = "images-resolution"
LR_FOLDER = "DIV2K_train_LR"
HR_FOLDER = "DIV2K_train_HR"
MODEL_OUTPUT_PATH = "s3://images-resolution/srcnn_model.pth"

# Initialize Spark session
spark = SparkSession.builder.appName("SRCNN Training").getOrCreate()

# Initialize S3 client
s3_client = boto3.client("s3")

# Define SRCNN model
class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.conv3(x)
        return x

# Load images from S3
def save_image_to_s3(img, s3_path):
    """ Saves a PIL image to S3 """
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    s3_client.put_object(Bucket=S3_BUCKET, Key=s3_path, Body=buffer)
    print(f"✅ Saved {s3_path} to S3")

def load_images_from_s3(lr_image_path, hr_image_path):
    try:
        print(f"📥 Loading images: {lr_image_path}, {hr_image_path}")

        # Load LR Image
        lr_response = s3_client.get_object(Bucket=S3_BUCKET, Key=lr_image_path)
        lr_img = Image.open(io.BytesIO(lr_response["Body"].read())).convert("RGB")

        # Load HR Image
        hr_response = s3_client.get_object(Bucket=S3_BUCKET, Key=hr_image_path)
        hr_img = Image.open(io.BytesIO(hr_response["Body"].read())).convert("RGB")

        # **Upload Processed Images to S3**
        save_image_to_s3(lr_img, f"processed_LR/{os.path.basename(lr_image_path)}")
        save_image_to_s3(hr_img, f"processed_HR/{os.path.basename(hr_image_path)}")

        return lr_img, hr_img
    except Exception as e:
        print(f"❌ Error loading images from S3: {e}")
        return None, None


# Preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Training function
def train_model():
    model = SRCNN()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Fetch image paths from S3
    lr_objects = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=LR_FOLDER).get("Contents", [])
    lr_image_paths = [obj["Key"] for obj in lr_objects if obj["Key"].endswith(".png")]

    hr_image_paths = [path.replace(LR_FOLDER, HR_FOLDER) for path in lr_image_paths]

    # Train on available images
    num_epochs = 5
    for epoch in range(num_epochs):
        total_loss = 0.0

        for lr_path, hr_path in zip(lr_image_paths, hr_image_paths):
            lr_img, hr_img = load_images_from_s3(lr_path, hr_path)

            if lr_img is None or hr_img is None:
                continue

            # Resize LR image to match HR dimensions
            lr_img = lr_img.resize(hr_img.size, Image.BICUBIC)

            # Convert to tensors
            lr_tensor = transform(lr_img).unsqueeze(0)
            hr_tensor = transform(hr_img).unsqueeze(0)

            # Forward pass
            output = model(lr_tensor)
            loss = criterion(output, hr_tensor)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss:.4f}")

    # Save trained model to S3
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    s3_client.put_object(Bucket=S3_BUCKET, Key="srcnn_model.pth", Body=buffer)

    print("✅ Model training completed and saved to S3!")

# Run training
train_model()

# Stop Spark session
spark.stop()
