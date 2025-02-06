#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status.
set -x  # Print each command before executing it.

echo "🚀 Bootstrapping AWS EMR for Super-Resolution Training..."

# Update and install required dependencies
echo "🔄 Updating package list and installing dependencies..."
sudo yum clean all
sudo yum update -y
sudo yum install -y python3 python3-pip awscli

# Install required Python dependencies in /tmp to avoid space issues
echo "📦 Installing Python dependencies..."
mkdir -p /tmp/python-packages
pip3 install --target=/tmp/python-packages --no-cache-dir torch torchvision boto3 tqdm opencv-python numpy pyspark matplotlib

# Ensure AWS S3 access
echo "🔑 Verifying AWS S3 access..."
aws s3 ls s3://images-resolution/ || echo "⚠️ Warning: Unable to access S3 bucket. Check IAM permissions."

echo "✅ Bootstrapping complete! Dependencies installed and environment configured."
