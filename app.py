import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Traffic Violation Detection", layout="centered")
st.title("🚦 Traffic Violation Detection")

VIOLATIONS = ["helmet_violation", "triple_riding"]

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    img = np.array(image)

    results = model(img)[0]
    violation = False

    for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
        label = model.names[int(cls)]
        if label in VIOLATIONS:
            violation = True
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    if violation:
        st.image(img, caption="Violation Detected", use_column_width=True)
    else:
        st.warning("No traffic violation detected")
