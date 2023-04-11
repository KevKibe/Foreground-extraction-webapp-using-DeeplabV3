
%%writefile app.py
import streamlit as st

# Streamlit app

from PIL import Image
from torchvision import transforms
import torch
from torch.nn import functional as F
import numpy as np
import warnings
import io
warnings.filterwarnings('ignore')

def extract_foreground(image_path):

    # Load DeepLabv3 model
    model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
    model.eval()

    # Load input image
    input_image = Image.open(image_path)

    # Get original image size
    original_size = input_image.size

    # Resize input image to a specific size
    image_size = (512, 512)  # specify desired image size
    input_image = input_image.resize(image_size)

    # Convert input image to RGB format
    input_image = input_image.convert("RGB")

    # Preprocess input image
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)

    # Perform inference
    with torch.no_grad():
        output_mask = model(input_batch)['out']
    output_mask = F.interpolate(output_mask, size=image_size, mode='bilinear', align_corners=False)
    output_mask = torch.argmax(output_mask, dim=1).squeeze().detach().cpu().numpy()

    # Create binary mask from output mask
    binary_mask = np.zeros_like(output_mask)
    binary_mask[output_mask == 15] = 1  # specify the class index for the foreground in DeepLabv3 (here, class index 15)

    # Overlay foreground on white canvas
    white_background = Image.new("RGB", input_image.size, (255, 255, 255))  # create a white canvas
    alpha = Image.fromarray(np.uint8(binary_mask * 255), mode="L")  # create an alpha channel from the binary mask
    foreground = Image.composite(input_image, white_background, alpha)

    # Resize foreground to original image size
    foreground = foreground.resize(original_size)

    return foreground

# Define CSS styles
PAGE_STYLE = """
<style>
body {
    font-family: Arial, sans-serif;
    background-color: #f8f9fa;
    margin: 0;
    padding: 0;
}

.container {
    max-width: 800px;
    margin: 0 auto;
    padding: 20px;
}

.title {
    font-size: 24px;
    font-weight: bold;
    margin-bottom: 20px;
}

.image {
    max-width: 100%;
    height: auto;
    margin-bottom: 20px;
}

.downloadButton {
    background-color: #007bff;
    color: #fff;
    padding: 12px 20px;
    border: none;
    cursor: pointer;
    font-size: 16px;
    margin-top: 16px;
}
.downloadButton:hover {
    background-color: #0056b3;
}
</style>
"""

# Create Streamlit app
st.title("Foreground Extraction")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width='auto')

    # Extract foreground
    extracted_foreground = extract_foreground(uploaded_file)
    st.image(extracted_foreground, caption="Extracted Foreground", use_column_width='auto')

    # Add download button for extracted foreground
    download_buffer = io.BytesIO()
    extracted_foreground.save(download_buffer, format='PNG')  # Change format to PNG
    download_button_str = f"Download Extracted Foreground"
    st.download_button(download_button_str, download_buffer.getvalue(), file_name='extracted_foreground.png')  # Change file extension to .png
