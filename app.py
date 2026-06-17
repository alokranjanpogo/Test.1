from ultralytics import YOLO
import streamlit as st
import cv2
import numpy as np
from PIL import Image

model = YOLO("best.pt")

uploaded_file = st.file_uploader(
    "Upload Intake Image",
    type=["jpg","jpeg","png"]
)

if uploaded_file:

    image = Image.open(uploaded_file)

    frame = np.array(image)

    results = model.predict(
        frame,
        conf=0.25,
        verbose=False
    )

    plotted = results[0].plot()

    st.image(
        plotted,
        caption="YOLO Result"
    )
