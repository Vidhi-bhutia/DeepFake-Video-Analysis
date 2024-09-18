import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.models import load_model
import tempfile
import os

# Load the pre-trained model
model = load_model('deepfake_video_detector.h5', compile=False)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Constants
IMG_SIZE = 224
MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2048

# Define feature extractor
def build_feature_extractor():
    feature_extractor = tf.keras.applications.InceptionV3(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    preprocess_input = tf.keras.applications.inception_v3.preprocess_input
    inputs = tf.keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)
    outputs = feature_extractor(preprocessed)
    return tf.keras.Model(inputs, outputs, name="feature_extractor")

feature_extractor = build_feature_extractor()

# Function to load video frames
def load_video(video_path, max_frames=MAX_SEQ_LENGTH, resize=(IMG_SIZE, IMG_SIZE)):
    cap = cv2.VideoCapture(video_path)
    frames = []
    try:
        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]  # Convert BGR to RGB
            frame = frame / 255.0  # Normalize pixel values
            frames.append(frame)
    finally:
        cap.release()
    return np.array(frames)

# Function to crop the center of the video frame
def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]

# Function to handle file upload and display
def handle_uploaded_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_file_path = temp_file.name

    frames = load_video(temp_file_path)
    return frames, temp_file_path

# Function to extract features from frames
def extract_features(frames):
    features = []
    for frame in frames:
        frame = np.expand_dims(frame, axis=0)
        feature = feature_extractor.predict(frame)
        features.append(feature[0])
    features = np.array(features)
    
    # Pad or truncate to MAX_SEQ_LENGTH
    if features.shape[0] > MAX_SEQ_LENGTH:
        features = features[:MAX_SEQ_LENGTH]
    elif features.shape[0] < MAX_SEQ_LENGTH:
        padding = np.zeros((MAX_SEQ_LENGTH - features.shape[0], NUM_FEATURES))
        features = np.vstack((features, padding))
    
    return features

# Function to predict if the video is fake or real
def predict_video(frames):
    features = extract_features(frames)
    features = np.expand_dims(features, axis=0)  # Shape: (1, MAX_SEQ_LENGTH, NUM_FEATURES)
    mask = np.ones((1, MAX_SEQ_LENGTH))  # Shape: (1, MAX_SEQ_LENGTH)
    prediction = model.predict([features, mask])
    return prediction[0][0]

# Streamlit app
st.title("DeepFake Video Detector")

uploaded_file = st.file_uploader("Choose a video file", type="mp4")
if uploaded_file:
    frames, temp_file_path = handle_uploaded_file(uploaded_file)
    
    if frames.size > 0:
        # Display the uploaded video
        st.video(temp_file_path)
        
        # Predict if the video is fake or real
        raw_prediction = predict_video(frames)
        
        # Display raw prediction score
        st.write(f"Raw prediction score: {raw_prediction}")
        
        # Adjust threshold if necessary
        threshold = 0.5  # You can adjust this value
        result = "Fake" if raw_prediction > threshold else "Real"
        
        # Display the result in the middle with large font
        st.markdown("<h1 style='text-align: center; color: red;'>Result</h1>", unsafe_allow_html=True)
        st.markdown(f"<h2 style='text-align: center; font-size: 48px;'>The video is predicted to be: {result}</h2>", unsafe_allow_html=True)
        
        # Remove the temporary file
        os.remove(temp_file_path)
    else:
        st.write("No frames could be extracted from the video.")

# Display model summary
st.write("Model Summary:")
model.summary(print_fn=lambda x: st.text(x))