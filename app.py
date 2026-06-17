import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

model = YOLO("best.pt")

uploaded = st.file_uploader("Upload")

if uploaded:
    img = Image.open(uploaded)
    frame = np.array(img)

    results = model.predict(
        frame,
        conf=0.25,
        verbose=False
    )

    for r in results:
    st.write("Masks:", r.masks is not None)

    if r.masks is not None:
        st.write("Number of masks:", len(r.masks.data))
