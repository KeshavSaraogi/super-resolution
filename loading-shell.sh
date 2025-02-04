#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status.
set -x  # Print each command before executing it.

# Logging setup
LOG_FILE="/var/log/dataproc_init.log"
exec > >(tee -a ${LOG_FILE}) 2>&1

echo "🚀 Initialization script started..."

# Update package list
echo "🔄 Updating package list..."
sudo apt-get update -y

# Install required Python dependencies
echo "🐍 Installing Python dependencies..."
sudo apt-get install -y python3-pip
pip install --upgrade pip
pip install --no-cache-dir torch torchvision torchaudio google-cloud-storage tqdm scikit-image

echo "✅ Setup and script execution complete!"
