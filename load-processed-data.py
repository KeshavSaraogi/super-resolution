from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import io
import torch
import boto3

# AWS S3 Settings
S3_BUCKET = "images-resolution"
HR_FOLDER = "DIV2K_train_HR"
LR_FOLDER = "DIV2K_train_LR_bicubic_X4"

# Initialize S3 Client
s3_client = boto3.client("s3")

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
])

class SRDataset(Dataset):
    def __init__(self, hr_paths, lr_paths, transform=None):
        self.hr_paths = hr_paths
        self.lr_paths = lr_paths
        self.transform = transform

    def __len__(self):
        return len(self.lr_paths)

    def __getitem__(self, idx):
        # Read images from S3
        hr_image = self.load_image(self.hr_paths[idx])
        lr_image = self.load_image(self.lr_paths[idx])

        if self.transform:
            hr_image = self.transform(hr_image)
            lr_image = self.transform(lr_image)

        return lr_image, hr_image

    def load_image(self, s3_path):
        """Loads an image from S3 bucket."""
        key = "/".join(s3_path.split("/")[3:])  # Extract S3 key
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
        image_bytes = response["Body"].read()
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")

# Load HR and LR image paths
hr_image_paths = [f"s3://{S3_BUCKET}/{HR_FOLDER}/{obj['Key'].split('/')[-1]}"
                  for obj in s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=HR_FOLDER)["Contents"]]

lr_image_paths = [f"s3://{S3_BUCKET}/{LR_FOLDER}/{obj['Key'].split('/')[-1]}"
                  for obj in s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=LR_FOLDER)["Contents"]]

# Create dataset and dataloader
dataset = SRDataset(hr_paths=hr_image_paths, lr_paths=lr_image_paths, transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

print("✅ Dataset Loaded Successfully! Number of samples:", len(dataset))
