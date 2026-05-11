import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

import streamlit as st
import keras
from PIL import Image, UnidentifiedImageError
import numpy as np
import json
import zipfile
import tempfile
import shutil


def _clean_layer_config(config):
    """Recursively strip keys not recognized by Keras 3.10 from layer configs."""
    unknown_keys = {'quantization_config', 'optional'}
    if isinstance(config, dict):
        cleaned = {}
        for k, v in config.items():
            if k in unknown_keys:
                continue
            cleaned[k] = _clean_layer_config(v)
        return cleaned
    elif isinstance(config, list):
        return [_clean_layer_config(item) for item in config]
    return config


def load_model_compat(model_path):
    """Load a .keras model saved with a newer Keras version by stripping
    unrecognized config fields before deserialization."""
    tmp_dir = tempfile.mkdtemp()
    try:
        with zipfile.ZipFile(model_path, 'r') as zf:
            zf.extractall(tmp_dir)

        config_path = os.path.join(tmp_dir, 'config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)

        config = _clean_layer_config(config)

        with open(config_path, 'w') as f:
            json.dump(config, f)

        # Re-pack into a temporary .keras file
        tmp_keras = os.path.join(tmp_dir, 'patched_model.keras')
        with zipfile.ZipFile(tmp_keras, 'w', zipfile.ZIP_DEFLATED) as zf_out:
            for root, dirs, files in os.walk(tmp_dir):
                for file in files:
                    if file == 'patched_model.keras':
                        continue
                    full_path = os.path.join(root, file)
                    arcname = os.path.relpath(full_path, tmp_dir)
                    zf_out.write(full_path, arcname)

        return keras.saving.load_model(tmp_keras, compile=False)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# Load model with error handling
try:
    model = load_model_compat('alzheimers_model.keras')
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

class_names = [
    'Mild Dementia',
    'Moderate Dementia',
    'Non Demented',
    'Very mild Dementia'
]

st.title("Alzheimer's Detection System")

# Add disclaimer
st.warning("**Disclaimer:** This tool is for educational/research purposes only and is not a substitute for professional medical diagnosis. Consult a healthcare provider for accurate assessment.")

# Option to use sample image or upload custom image
col1, col2 = st.columns(2)
use_sample = False

with col1:
    if st.button("Use Sample Image"):
        use_sample = True

with col2:
    st.write("or")

uploaded_file = st.file_uploader(
    "Upload an MRI Image",
    type=['jpg', 'jpeg', 'png']
)

# Use uploaded file if available, otherwise check for sample image button
if use_sample and not uploaded_file:
    try:
        uploaded_file = open("Screenshot 2026-05-10 232500.png", "rb")
    except FileNotFoundError:
        st.error("Sample image not found.")

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
        st.image(image, caption='Uploaded MRI Image', use_container_width=True)

        # Resize and preprocess
        image = image.resize((224, 224))
        img_array = np.array(image) / 255.0
        img_array = np.stack([img_array] * 3, axis=-1)  # Duplicate to 3 channels
        img_array = img_array.astype(np.float32)
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction with spinner
        with st.spinner('Analyzing image...'):
            prediction = model.predict(img_array)

        predicted_class = class_names[np.argmax(prediction[0])]
        confidence = np.max(prediction[0]) * 100

        st.subheader(f"Prediction: {predicted_class}")
        st.write(f"Confidence: {confidence:.2f}%")

        # Show raw predictions
        st.subheader("Raw Prediction Probabilities")
        for i, prob in enumerate(prediction[0]):
            st.write(f"{class_names[i]}: {prob * 100:.2f}%")

    except UnidentifiedImageError:
        st.error("Invalid image file. Please upload a valid JPG, JPEG, or PNG image.")
    except Exception as e:
        st.error(f"An error occurred during processing: {e}")