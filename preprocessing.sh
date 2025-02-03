#!/bin/bash

echo "🚀 Running Initialization Action for Dataproc Cluster Setup..."
pip3 install pyspark google-cloud-storage opencv-python pillow numpy tensorflow

# ---------------------- STEP 3: Verify Installations ----------------------
echo "✅ Verifying installations..."
python3 -c "import pyspark; print('PySpark Installed')"
python3 -c "import cv2; print('OpenCV Installed')"

echo "🎉 Initialization Action Completed!"
