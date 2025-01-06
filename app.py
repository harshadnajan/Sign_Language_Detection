import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

# Set up MediaPipe hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.3)

# Streamlit video input widget
st.title("Sign Language Recognition")
st.subheader("Use your webcam to recognize signs")

video_capture = st.camera_input("Capture Video")

if video_capture:
    # Convert the PIL image to a NumPy array
    frame = np.array(video_capture)

    # Check if the frame is valid (non-empty)
    if frame is not None and frame.size > 0:
        # Convert the image from RGB to BGR (since OpenCV uses BGR format)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Process the frame with MediaPipe
        results = hands.process(frame_bgr)

        # Draw landmarks if any hands are detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame_bgr, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Display the frame with landmarks
        st.image(frame_bgr, channels="BGR")
    else:
        st.error("Failed to capture frame from webcam.")
