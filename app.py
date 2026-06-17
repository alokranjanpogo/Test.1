import streamlit as st
import os

st.write("Files in directory:")
st.write(os.listdir("."))

st.write("best.pt exists:")
st.write(os.path.exists("best.pt"))
