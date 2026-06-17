import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

model = YOLO("best.pt")

uploaded = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded:
    image = Image.open(uploaded)
    img = np.array(image)

    results = model.predict(
        img,
        conf=0.25,
        verbose=False
    )

    st.image(
        results[0].plot(),
        caption="Prediction"
    )
