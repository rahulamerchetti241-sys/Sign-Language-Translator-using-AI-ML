import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# ---------------- CONFIG ----------------
STABILITY_FRAMES = 20
CONFIDENCE_THRESHOLD = 0.85

# ---------------- LOAD MODEL ----------------
# Ensure these files exist in your directory
try:
    model = tf.keras.models.load_model("sign_language_model.keras")
    with open("label_encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# ---------------- MEDIAPIPE ----------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

st.set_page_config(page_title="AI Sign Language Translator")
st.title("AI Sign Language Translator (ASL)")

# ---------------- VIDEO PROCESSOR ----------------
class SignRecognizer(VideoTransformerBase):
    def __init__(self):
        self.sentence = []
        self.last_pred = None
        self.frame_counter = 0

    def transform(self, frame):
        image = frame.to_ndarray(format="bgr24")
        image = cv2.flip(image, 1)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process Hand
        results = hands.process(rgb)
        
        predicted_char = "Waiting..."
        hand_detected = False

        if results.multi_hand_landmarks:
            hand_detected = True
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw thicker landmarks for better visibility
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                )

                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y])

                input_data = np.array([landmarks])
                prediction = model.predict(input_data, verbose=0)
                class_id = np.argmax(prediction)
                confidence = np.max(prediction)

                if confidence > CONFIDENCE_THRESHOLD:
                    predicted_char = encoder.inverse_transform([class_id])[0]

                    if predicted_char == self.last_pred:
                        self.frame_counter += 1
                    else:
                        self.frame_counter = 0
                        self.last_pred = predicted_char

                    if self.frame_counter == STABILITY_FRAMES:
                        p = predicted_char.lower()
                        if "space" in p:
                            self.sentence.append(" ")
                        elif "delete" in p:
                            if self.sentence:
                                self.sentence.pop()
                        else:
                            self.sentence.append(predicted_char)
                        self.frame_counter = 0

        # ---------------- VISUALIZATION UI ----------------
        h, w, _ = image.shape
        
        # 1. Create a semi-transparent background for text (Header Bar)
        # Covers top 100 pixels
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (w, 100), (0, 0, 0), -1) 
        alpha = 0.7  # Transparency level
        image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

        # 2. Add Status Indicator (Circle in top right)
        status_color = (0, 255, 0) if hand_detected else (0, 0, 255) # Green if hand, Red if not
        cv2.circle(image, (w - 30, 30), 15, status_color, -1)
        cv2.circle(image, (w - 30, 30), 17, (255, 255, 255), 2) # White border

        # 3. Display Current Prediction (Big and Visible)
        cv2.putText(image, "DETECTING:", (20, 35), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA)
        
        # Display the character in Bright Yellow
        disp_char = predicted_char if hand_detected else "..."
        cv2.putText(image, f"{disp_char}", (160, 35), 
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)

        # 4. Display Constructed Sentence
        text_str = "".join(self.sentence)
        cv2.putText(image, "SENTENCE:", (20, 80), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA)
        
        # Display the sentence in Bright Green
        # Show only last 25 chars to prevent overflow
        cv2.putText(image, f"{text_str[-25:]}", (160, 80), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        # 5. Stability Progress Bar (Optional: Shows how close to locking in a letter)
        if hand_detected and self.last_pred:
            bar_width = int((self.frame_counter / STABILITY_FRAMES) * 100)
            cv2.rectangle(image, (w - 140, 60), (w - 40, 70), (100, 100, 100), -1) # Background
            cv2.rectangle(image, (w - 140, 60), (w - 140 + bar_width, 70), (0, 255, 255), -1) # Fill

        return image

# Start Streamer
webrtc_streamer(
    key="sign-language",
    video_transformer_factory=SignRecognizer,
    media_stream_constraints={"video": True, "audio": False}
)