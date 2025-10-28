import streamlit as st
import cv2
from fer.fer import FER
from PIL import Image
import numpy as np
import time

st.set_page_config(page_title="Emotion Detection", layout="centered")
st.title("🎭 Real-Time Emotion Detection using Streamlit")

st.write("This app detects your emotion live using your webcam!")

# Initialize emotion detector
detector = FER(mtcnn=True)

# Start camera
run = st.checkbox('Start Camera')
FRAME_WINDOW = st.image([])

camera = cv2.VideoCapture(0)

while run:
    ret, frame = camera.read()
    if not ret:
        st.warning("No camera detected!")
        break

    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect emotion
    result = detector.detect_emotions(rgb_frame)

    # Draw bounding box and emotion
    for face in result:
        (x, y, w, h) = face["box"]
        emotion, score = max(face["emotions"].items(), key=lambda x: x[1])
        cv2.rectangle(rgb_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(rgb_frame, f"{emotion} ({score:.2f})", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)

    FRAME_WINDOW.image(rgb_frame)
    time.sleep(0.05)

else:
    st.write("Camera stopped.")
    camera.release()
