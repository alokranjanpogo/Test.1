import streamlit as st
from ultralytics import YOLO

st.write("Before loading")

try:
    model = YOLO("best.pt")
    st.success("Model Loaded Successfully")

except Exception as e:
    st.error(str(e))
