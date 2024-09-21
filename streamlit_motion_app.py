import streamlit as st
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from datetime import datetime

# Function to spot differences
def spot_diff(frame1, frame2):
    g1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    g1 = cv2.blur(g1, (2, 2))
    g2 = cv2.blur(g2, (2, 2))

    (score, diff) = ssim(g2, g1, full=True)
    diff = (diff * 255).astype("uint8")
    thresh = cv2.threshold(diff, 100, 255, cv2.THRESH_BINARY_INV)[1]

    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    contours = [c for c in contours if cv2.contourArea(c) > 50]

    if len(contours):
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return frame1

# Function to detect motion
def detect_motion():
    st.title("Motion Detection App")
    st.write("Starting camera...")

    cap = cv2.VideoCapture(0)
    time.sleep(2)  # Allow the camera to warm up

    if not cap.isOpened():
        st.error("Error: Camera not accessible.")
        return

    ret, frame1 = cap.read()
    if not ret:
        st.error("Error: Unable to read frame from camera.")
        cap.release()
        return

    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    while True:
        ret, frame2 = cap.read()
        if not ret:
            st.error("Error: Unable to read frame from camera.")
            break

        frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        diff_frame = spot_diff(frame1, frame2)
        
        # Display the result
        st.image(diff_frame, channels="BGR", caption="Detected Motion")
        
        frame1 = frame2_gray

        if st.button("Stop"):
            break

    cap.release()
    cv2.destroyAllWindows()

# Streamlit UI
if st.button("Start Motion Detection"):
    detect_motion()
