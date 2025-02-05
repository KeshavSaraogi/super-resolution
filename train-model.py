import torch
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import os
from google.cloud import storage
from srcnn_model import SRCNN  # Import the model

# Load Dataset
def get_data_loader(batch_size=16):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    
    dataset_path = "/home/dataproc/dataset"
    
    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader

# Train Model
def train_model(model, dataloader, epochs=10, lr=0.001):
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    
    for epoch in range(epochs):
        running_loss = 0.0
        for images, _ in dataloader:
            images = images.cuda() if torch.cuda.is_available() else images
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss:.4f}")
    
    return model

# Save Model to GCS
def save_model_to_gcs(model, gcs_bucket, model_filename):
    client = storage.Client()
    bucket = client.get_bucket(gcs_bucket)
    
    model_path = "/home/dataproc/" + model_filename
    torch.save(model.state_dict(), model_path)
    
    blob = bucket.blob(model_filename)
    blob.upload_from_filename(model_path)
    
    print(f"✅ Model saved to GCS: gs://{gcs_bucket}/{model_filename}")

# Main Execution
if __name__ == "__main__":
    model = SRCNN()
    model = model.cuda() if torch.cuda.is_available() else model
    
    dataloader = get_data_loader()
    trained_model = train_model(model, dataloader)
    
    save_model_to_gcs(trained_model, "super-resolution-images", "super_res_model.pth")
    
    print("🎉 Model training complete and uploaded to GCS!")
