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

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    st.subheader("📷 Uploaded Image")
    st.image(image, use_column_width=True)

    with st.spinner("🔍 Detecting violations..."):
        results = model(img_array)

    annotated_image = results[0].plot()

    st.subheader("🚨 Detected Violations")
    st.image(annotated_image, use_column_width=True)

    if results[0].boxes is not None and len(results[0].boxes) > 0:
        st.subheader("📊 Detection Details")
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]
            st.write(f"🔹 **{label}** — Confidence: `{conf:.2f}`")
    else:
        st.info("No traffic violations detected.")
