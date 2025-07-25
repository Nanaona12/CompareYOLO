import streamlit as st
from ultralytics import YOLO
from PIL import Image
import torch
import os
import time
import torch

model_v9 = YOLO("pretrained model/trainv9/weights/best.pt")
model_v10 = YOLO("pretrained model/trainv10/weights/best.pt")
model_v11 = YOLO("pretrained model/trainv11/weights/best.pt")

def detect_objects(model, image):
    start_time = time.time()
    results = model(image)
    inference_time = time.time() - start_time
    return results[0].plot(), inference_time

st.title("YOLOv9, YOLOv10, YOLOv11 Image Detection and Performance Comparison")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    st.write("Processing...")
    
    # Perform detection with each model and measure performance
    image_v9, time_v9 = detect_objects(model_v9, image)
    image_v10, time_v10 = detect_objects(model_v10, image)
    image_v11, time_v11 = detect_objects(model_v11, image)
    
    # Display results
    st.image(image_v9, caption=f"YOLOv9 Detection Result (Time: {time_v9:.2f}s)", use_column_width=True)
    st.image(image_v10, caption=f"YOLOv10 Detection Result (Time: {time_v10:.2f}s)", use_column_width=True)
    st.image(image_v11, caption=f"YOLOv11 Detection Result (Time: {time_v11:.2f}s)", use_column_width=True)
    
    # Display comparison table
    st.write("### Model Performance Comparison")
    st.table({
        "Model": ["YOLOv9", "YOLOv10", "YOLOv11"],
        "Inference Time (s)": [time_v9, time_v10, time_v11]
    })

st.write("Upload an image to start detection!")

