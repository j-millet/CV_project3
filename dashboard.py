import streamlit as st
from main import setup_model, get_transform, TARGET_SIZE,THRESHOLD
from utils import get_output_images, collate_fn
import numpy as np
from PIL import Image
from dataset import InferenceDataset
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F




@st.cache_resource
def load_model(name):
    if name == "Mask R-CNN":
        return setup_model('data/model_checkpoints/mask_rcnn_model.pth')
    elif name == "Custom1":
        return setup_model("data/model_checkpoints/custom1_model.pth")
    elif name == "Custom2":
        return setup_model("data/model_checkpoints/custom2_model.pth")


st.title("Cityscapes Mask R-CNN Dashboard")
model_num = st.selectbox("Choose model", ["Mask R-CNN", "Custom1", "Custom2"])
model = load_model(model_num)


@st.cache_data
def run_inference(imfile):
    image = Image.open(imfile).convert("RGB")
    dataset = InferenceDataset(
        [image], transforms=get_transform(False), target_size=TARGET_SIZE
    )
    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, pin_memory=True
    )
    inf = get_output_images(model, dataloader, next(model.parameters()).device, THRESHOLD)[0]
    inf_img = F.to_pil_image(inf.detach().cpu())
    inf_img_resized = inf_img.resize(image.size)
    return inf_img_resized


st.write("Upload an image:")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    inf_button = st.button("Run Inference")

    # Do split view columns
    col1, col2 = st.columns(2)
    with col1:
        st.image(uploaded_file, caption="Uploaded Image.", use_container_width=True)
    with col2:
        if inf_button:
            image = run_inference(uploaded_file)
            st.image(image, caption="Inference Result.", use_container_width=True)
