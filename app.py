import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os

st.set_page_config(page_title="Traffic Violation Detection", layout="centered")

st.title("🚦 Traffic Violation Detection System")
st.write("Vehicles are highlighted only when traffic violations are detected.")

MODEL_PATH = "best.pt"

if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file (best.pt) not found.")
    st.stop()

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

# ⚠️ UPDATE CLASS NAMES TO MATCH YOUR DATASET
VEHICLE_CLASSES = ["motorcycle", "bike"]
VIOLATION_CLASSES = ["helmet_violation", "triple_riding"]

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    boxBArea = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])

    return interArea / (boxAArea + boxBArea - interArea + 1e-6)

def detect_violations(image_np):
    results = model(image_np, conf=0.4)
    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        return None

    vehicles = []
    violations = []

    for box in boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        xyxy = box.xyxy[0].cpu().numpy()

        if label in VEHICLE_CLASSES:
            vehicles.append((box, xyxy))
        elif label in VIOLATION_CLASSES:
            violations.append((box, xyxy))

    if len(vehicles) == 0 or len(violations) == 0:
        return None

    final_vehicle_boxes = []

    for v_box, v_xyxy in vehicles:
        for viol_box, viol_xyxy in violations:
            if iou(v_xyxy, viol_xyxy) > 0.3:
                final_vehicle_boxes.append(v_box)
                break

    if len(final_vehicle_boxes) == 0:
        return None

    results[0].boxes = final_vehicle_boxes
    return results[0].plot()

uploaded_file = st.file_uploader(
    "Upload Traffic Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    st.subheader("Uploaded Image")
    st.image(image, use_column_width=True)

    with st.spinner("Analyzing traffic scene..."):
        output = detect_violations(image_np)

    if output is None:
        st.success("✅ No traffic violations detected.")
    else:
        st.subheader("🚨 Traffic Violation Detected")
        st.image(output, use_column_width=True)
