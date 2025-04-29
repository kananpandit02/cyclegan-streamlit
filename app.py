import streamlit as st
import torch
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import os
import tempfile
from models import Generator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load models
@st.cache_resource
def load_models():
    G_AB = Generator().to(device)
    G_BA = Generator().to(device)

    checkpoint = torch.load('cyclegan_epoch_50.pth', map_location=device)
    G_AB.load_state_dict(checkpoint['G_AB_state_dict'])
    G_BA.load_state_dict(checkpoint['G_BA_state_dict'])

    G_AB.eval()
    G_BA.eval()
    return G_AB, G_BA

G_AB, G_BA = load_models()

# Image transform
transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Translate an image
def translate_image(image, direction='A2B'):
    img = transform_test(image).unsqueeze(0).to(device)
    with torch.no_grad():
        if direction == 'A2B':
            fake_img = G_AB(img)
        else:
            fake_img = G_BA(img)
    fake_img = (fake_img + 1) / 2  # Denormalize
    return fake_img.squeeze(0).cpu()

# Streamlit UI
st.title("CycleGAN Image Translator By Kanan Pandit and Partha Mete @RKMVERI 🚀")
st.write("Upload an image and choose translation direction:")

uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])
direction = st.radio("Select Direction:", ('REAL To GHIBLI', 'GHIBLI To REAL'))

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Translate"):
        output_tensor = translate_image(image, direction)
        output_image_path = os.path.join(tempfile.gettempdir(), "translated.png")
        save_image(output_tensor, output_image_path)
        st.image(output_image_path, caption="Translated Image", use_column_width=True)

        with open(output_image_path, "rb") as f:
            st.download_button("Download Result", f, file_name="translated.png", mime="image/png")
