#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status.

pip install torch torchvision torchaudio google-cloud-storage tqdm scikit-image

echo "✅ Setup and script execution complete!"
