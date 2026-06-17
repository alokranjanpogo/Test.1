import streamlit as st
from ultralytics import YOLO

model = YOLO("best.pt")

st.success("Model Loaded")

results = model.predict(
    "https://ultralytics.com/images/bus.jpg",
    verbose=False
)

st.write(results)
st.success("Prediction Completed")
