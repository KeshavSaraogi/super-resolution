import streamlit as st
import torch
import torchvision.transforms as transforms
import boto3
import io
from PIL import Image
from model import SRCNN  # Import the trained model class

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

# **✅ Streamlit UI Enhancements**
st.set_page_config(page_title="Super-Resolution App", layout="wide")

st.sidebar.title("⚙️ Options")
st.sidebar.write("Upload a low-resolution image to enhance it.")

uploaded_files = st.sidebar.file_uploader("Choose images...", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    col1, col2 = st.columns(2)
    
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).convert("RGB")

        with col1:
            st.image(image, caption="Original Image", use_column_width=True)

        with col2:
            with st.spinner("🛠️ Enhancing image..."):
                enhanced_image = process_image(image)
                st.image(enhanced_image, caption="Enhanced Image", use_column_width=True)
                
                buf = io.BytesIO()
                enhanced_image.save(buf, format="PNG")
                buf.seek(0)
                st.download_button("📥 Download Enhanced Image", buf, "enhanced_image.png", "image/png")
