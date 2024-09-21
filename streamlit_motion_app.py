import streamlit as st
import cv2
import numpy as np
import time
from datetime import datetime
from skimage.metrics import structural_similarity as compare_ssim

# Function to find and highlight differences
def spot_diff(frame1, frame2):
    g1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    g1 = cv2.blur(g1, (2, 2))
    g2 = cv2.blur(g2, (2, 2))

    (score, diff) = compare_ssim(g1, g2, full=True)

    diff = (diff * 255).astype("uint8")
    thresh = cv2.threshold(diff, 100, 255, cv2.THRESH_BINARY_INV)[1]

    contors = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    contors = [c for c in contors if cv2.contourArea(c) > 50]

    if len(contors):
        for c in contors:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return True, frame1
    return False, frame1

# Function to detect motion
def detect_motion():
    cap = cv2.VideoCapture(0)
    time.sleep(2)
    
    ret, frame1 = cap.read()
    frm1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    
    motion_detected = False
    is_start_done = False
    start_time = None

    while True:
        ret, frame2 = cap.read()
        frm2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        diff = cv2.absdiff(frm1, frm2)
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

        contors = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        contors = [c for c in contors if cv2.contourArea(c) > 50]

        if len(contors) > 5:
            motion_detected = True
            st.write("Motion Detected!")

        elif motion_detected and len(contors) < 3:
            if not is_start_done:
                start_time = time.time()
                is_start_done = True
            
            if time.time() - start_time > 4:
                ret, new_frame = cap.read()
                cap.release()

                stolen_detected, result_frame = spot_diff(frame1, new_frame)
                if stolen_detected:
                    st.write("Object stolen detected!")
                    st.image(result_frame, channels="BGR")
                    st.write("Saving image...")
                    cv2.imwrite("stolen/" + datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + ".jpg", result_frame)
                else:
                    st.write("No object stolen. Restarting detection.")
                return
        
        st.image(frame2, channels="BGR")
        time.sleep(0.1)
        frm1 = frm2

# Streamlit UI
st.title("Theft Detection App")
st.write("Click the button below to start detecting motion.")

if st.button("Start Motion Detection"):
    detect_motion()

st.write("Theft detection system using webcam. If motion is detected, the system will try to identify if an object is stolen.")
