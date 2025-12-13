import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.title("🚦 Traffic Violation Detection System")

model = YOLO("best.pt")

uploaded_file = st.file_uploader("Upload an image", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")

    results = model(image)
    st.write("### Detected Violations")
    st.write(results[0].boxes)
