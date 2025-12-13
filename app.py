import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os

st.set_page_config(page_title="Traffic Violation Detection", layout="centered")

st.title("🚦 Traffic Violation Detection System")
st.write("Upload an image to detect traffic violations using YOLOv8")

MODEL_PATH = "best.pt"

if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file best.pt not found")
    st.stop()

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

uploaded_file = st.file_uploader(
    "Upload Traffic Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is_
boxes)
