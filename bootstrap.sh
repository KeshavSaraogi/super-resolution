#!/bin/bash
set -e

echo "Updating system packages..."
sudo yum update -y

echo "Installing Python and required dependencies..."
sudo yum install -y python3 pip git
python3 -m pip install --upgrade pip
python3 -m pip install torch torchvision boto3 pillow pyspark

echo "Setting up environment variables..."
echo "export PATH=/usr/bin:$PATH" >> ~/.bashrc
echo "alias python=python3" >> ~/.bashrc
source ~/.bashrc

echo "Bootstrap script execution completed!"