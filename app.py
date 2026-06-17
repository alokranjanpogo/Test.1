import streamlit as st
import cv2
from ultralytics import YOLO

st.success("OpenCV Loaded")
st.success(f"OpenCV Version: {cv2.__version__}")

model = YOLO("best.pt")

st.success("YOLO Loaded Successfully")
