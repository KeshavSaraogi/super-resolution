#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status.
set -x  # Print each command before executing it.

# Logging setup
LOG_FILE="/var/log/model_training.log"
exec > >(tee -a ${LOG_FILE}) 2>&1

echo "🚀 Model training initialization started..."

# Update package list and install dependencies
echo "🔄 Updating package list..."
sudo apt-get update -y
sudo apt-get install -y python3-pip

# Install Python dependencies
echo "🐍 Installing required Python libraries..."
pip install --upgrade pip
pip install torch torchvision torchaudio google-cloud-storage tqdm scikit-image numpy pandas matplotlib

# Download dataset from GCS
echo "📥 Downloading dataset..."
gsutil -m cp -r gs://super-resolution-images/dataset /home/dataproc/

# Run the model training script
echo "🎯 Running model training..."
python3 /home/dataproc/model-training.py

echo "✅ Model training completed!"
