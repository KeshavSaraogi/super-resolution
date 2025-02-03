from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import io
import torch
from google.cloud import storage

# Google Cloud Storage settings
GCS_BUCKET = "super-resolution-images"
HR_FOLDER = "DIV2K_train_HR"
LR_FOLDER = "DIV2K_train_LR"

# Initialize GCS Client
storage_client = storage.Client()

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
        # Read images from GCS
        hr_image = self.load_image(self.hr_paths[idx])
        lr_image = self.load_image(self.lr_paths[idx])

        if self.transform:
            hr_image = self.transform(hr_image)
            lr_image = self.transform(lr_image)

        return lr_image, hr_image

    def load_image(self, gcs_path):
        """Loads an image from GCS."""
        bucket_name = gcs_path.split('/')[2]
        blob_path = '/'.join(gcs_path.split('/')[3:])
        
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        image_bytes = blob.download_as_bytes()

        return Image.open(io.BytesIO(image_bytes)).convert("RGB")

# Load HR and LR image paths
hr_image_paths = [f"gs://{GCS_BUCKET}/{HR_FOLDER}/{blob.name.split('/')[-1]}" 
                  for blob in storage_client.bucket(GCS_BUCKET).list_blobs(prefix=HR_FOLDER)]

lr_image_paths = [f"gs://{GCS_BUCKET}/{LR_FOLDER}/{blob.name.split('/')[-1]}" 
                  for blob in storage_client.bucket(GCS_BUCKET).list_blobs(prefix=LR_FOLDER)]

# Create dataset and dataloader
dataset = SRDataset(hr_paths=hr_image_paths, lr_paths=lr_image_paths, transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

print("✅ Dataset Loaded Successfully! Number of samples:", len(dataset))
