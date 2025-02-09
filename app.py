import streamlit as st
import torch
import torchvision.transforms as transforms
import boto3
import io
from PIL import Image
from model import SRCNN  # Import your trained model class

# **✅ Load Trained Model from S3**
S3_BUCKET = "images-resolution"
MODEL_PATH = "srcnn_model.pth"

s3_client = boto3.client("s3")
@st.cache_resource
def load_model():
    buffer = io.BytesIO()
    s3_client.download_fileobj(S3_BUCKET, MODEL_PATH, buffer)
    buffer.seek(0)

    model = SRCNN()
    model.load_state_dict(torch.load(buffer, map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

# **✅ Image Processing Function**
def process_image(image):
    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)
    
    output_img = transforms.ToPILImage()(output.squeeze(0))
    return output_img

# **✅ Streamlit UI**
st.title("🖼️ Super-Resolution Image Enhancer")
st.write("Upload a low-resolution image and enhance it using the trained SRCNN model.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original Image", use_column_width=True)
    
    if st.button("Enhance Image"):
        enhanced_image = process_image(image)
        st.image(enhanced_image, caption="Enhanced Image", use_column_width=True)
        
        # Allow users to download the enhanced image
        buf = io.BytesIO()
        enhanced_image.save(buf, format="PNG")
        buf.seek(0)
        st.download_button("Download Enhanced Image", buf, "enhanced_image.png", "image/png")
