import streamlit as st
from main import get_transform, TARGET_SIZE,THRESHOLD, DEVICE
from utils import get_output_images, collate_fn, load_model_for_inference
from PIL import Image
from dataset import InferenceDataset
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F


@st.cache_resource
def load_model(name):
    model_name = "mask_rcnn" if name == "Mask R-CNN" else "instance_seg"
    path = "data/model_checkpoints/mask_rcnn_model.pth" if model_name == "mask_rcnn" else "data/model_checkpoints/InstSegNet_adamw_model.pth"
    return load_model_for_inference(model_name,path, DEVICE)


st.title("Cityscapes Mask R-CNN Dashboard")
model_name = st.selectbox("Choose model", ["Mask R-CNN", "Simple Instance Segmentation"])
model = load_model(model_name)


@st.cache_data
def run_inference(imfile,mname):
    print(mname)
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
            image = run_inference(uploaded_file,model_name)
            st.image(image, caption="Inference Result.", use_container_width=True)
