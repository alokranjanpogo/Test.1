from ultralytics import YOLO
import streamlit as st
import cv2
import numpy as np
from PIL import Image

model = YOLO("best.pt")
def cleaning_frequency(load_percent):

    if load_percent < 5:
        return "Weekly Cleaning", "Low"

    elif load_percent < 15:
        return "Twice a Week", "Moderate"

    elif load_percent < 30:
        return "Every Alternate Day", "Medium"

    elif load_percent < 50:
        return "Daily Cleaning", "High"

    else:
        return "Immediate Cleaning Required", "Critical"
uploaded_file = st.file_uploader(
    "Upload Intake Image",
    type=["jpg","jpeg","png"]
)

if uploaded_file:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Intake Image",
        use_container_width=True
    )

    if st.button("🚀 Run Diagnosis"):

        with st.spinner(
            "AI Model Analyzing Plastic Accumulation..."
        ):

            frame = np.array(image)

            if len(frame.shape) == 2:
                frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_GRAY2RGB
                )

            frame_bgr = cv2.cvtColor(
                frame,
                cv2.COLOR_RGB2BGR
            )

            h, w = frame_bgr.shape[:2]

            results = model.predict(
                frame_bgr,
                conf=0.25,
                verbose=False
            )

            # ==========================
            # DISPLAY SEGMENTATION
            # ==========================

            plotted = results[0].plot()

            # ==========================
            # AREA CALCULATION
            # ==========================

            plastic_pixels = 0

            for result in results:

                if result.masks is not None:

                    masks = result.masks.data.cpu().numpy()

                    for mask in masks:

                        mask = cv2.resize(
                            mask,
                            (w, h)
                        )

                        binary_mask = (
                            mask > 0.5
                        ).astype(np.uint8)

                        plastic_pixels += np.sum(
                            binary_mask
                        )

            total_pixels = h * w

            load_percent = (
                plastic_pixels /
                total_pixels
            ) * 100

            recommendation, risk = cleaning_frequency(
                load_percent
            )

        st.success(
            "Diagnosis Completed Successfully"
        )

        st.image(
            plotted,
            caption="AI Segmentation Result",
            use_container_width=True
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Plastic Load (%)",
                f"{load_percent:.2f}"
            )

        with col2:
            st.metric(
                "Risk Level",
                risk
            )

        with col3:
            st.metric(
                "Plastic Pixels",
                f"{plastic_pixels:,}"
            )

        st.subheader(
            "🧹 Cleaning Recommendation"
        )

        st.success(
            f"Recommended Frequency: {recommendation}"
        )

        if load_percent > 50:

            st.error(
                "🚨 Critical Plastic Accumulation Detected. Immediate Cleaning Recommended."
            )

        elif load_percent > 30:

            st.warning(
                "⚠ High Plastic Accumulation Detected."
            )

        elif load_percent > 15:

            st.info(
                "🔍 Moderate Plastic Accumulation Detected."
            )

        else:

            st.success(
                "✅ Plastic Load Within Acceptable Range."
            )

        with st.expander(
            "Technical Details"
        ):

            st.write(
                "Image Resolution:",
                f"{w} x {h}"
            )

            st.write(
                "Plastic Pixels:",
                plastic_pixels
            )

            st.write(
                "Total Pixels:",
                total_pixels
            )

            st.write(
                "Load Percentage:",
                round(load_percent, 2)
            )

            st.write(
                "Model Classes:",
                model.names
            )
