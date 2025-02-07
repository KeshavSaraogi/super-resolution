import os
import boto3
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
from io import BytesIO

# AWS S3 CONFIGURATION
S3_BUCKET = "images-resolution"
S3_IMAGE_PREFIX = "train-images/"  # Folder where images are stored
S3_MODEL_SAVE_PATH = "trained-model.pth"

# LOCAL DIRECTORY FOR STORING IMAGES
LOCAL_IMAGE_DIR = "/tmp/train-images"
os.makedirs(LOCAL_IMAGE_DIR, exist_ok=True)

# Initialize AWS S3 Client
s3_client = boto3.client("s3")

def ensure_s3_folder_exists(bucket, prefix):
    """Ensure that a given folder exists in the S3 bucket."""
    result = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
    if "Contents" not in result:
        print(f"Creating missing folder: {prefix}")
        s3_client.put_object(Bucket=bucket, Key=f"{prefix}dummy.txt", Body="Placeholder")

def download_images_from_s3():
    """Download training images from S3 and save them locally."""
    response = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=S3_IMAGE_PREFIX)
    
    if "Contents" not in response:
        raise ValueError("No training images found in S3 bucket.")

    for obj in response["Contents"]:
        file_key = obj["Key"]
        if file_key.endswith((".jpg", ".png", ".jpeg")):
            print(f"Downloading {file_key} ...")
            file_obj = s3_client.get_object(Bucket=S3_BUCKET, Key=file_key)
            image = Image.open(BytesIO(file_obj["Body"].read()))
            image.save(os.path.join(LOCAL_IMAGE_DIR, os.path.basename(file_key)))

def get_transforms():
    """Return image transformations for training."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

def load_dataset():
    """Load images and apply transformations."""
    transform = get_transforms()
    images = []
    labels = []  # Assuming binary classification for simplicity
    
    for file in os.listdir(LOCAL_IMAGE_DIR):
        if file.endswith((".jpg", ".png", ".jpeg")):
            img_path = os.path.join(LOCAL_IMAGE_DIR, file)
            image = Image.open(img_path).convert("RGB")
            images.append(transform(image))
            labels.append(0)  # Default class label (update this based on your dataset)

    return torch.stack(images), torch.tensor(labels, dtype=torch.long)

class SimpleCNN(nn.Module):
    """Basic CNN model for training."""
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)  # Binary classification

    def forward(self, x):
        return self.model(x)

def train_model():
    """Train a simple CNN model."""
    ensure_s3_folder_exists(S3_BUCKET, S3_IMAGE_PREFIX)
    download_images_from_s3()
    
    images, labels = load_dataset()
    dataset = torch.utils.data.TensorDataset(images, labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 5
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader)}")

    # Save trained model locally
    model_path = "/tmp/trained-model.pth"
    torch.save(model.state_dict(), model_path)

    # Upload trained model to S3
    s3_client.upload_file(model_path, S3_BUCKET, S3_MODEL_SAVE_PATH)
    print(f"Model saved to S3: {S3_MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_model()
