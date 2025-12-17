# Sign-Language-Translator-using-AI-ML
AI-based Sign Language Translator that recognizes hand gestures in real time using MediaPipe and a trained Keras model. Hand landmarks are extracted from live webcam input and classified to display text output. The system is deployed as a browser-based web application using WebRTC for online access.

A real-time AI-powered Sign Language Translator built using:
- Python
- TensorFlow
- MediaPipe
- OpenCV
- Streamlit + WebRTC

## Features
- Real-time hand tracking
- ASL alphabet recognition
- Sentence construction
- Confidence-based prediction
- Clean UI with live camera feed

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py

