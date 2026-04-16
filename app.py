import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

# ===============================
# LOAD MODEL + CLASSES
# ===============================
model = tf.keras.models.load_model("models/model.h5")

with open("models/classes.json", "r") as f:
    class_names = json.load(f)

# ===============================
# UI
# ===============================
st.title("♻️ Garbage Classification System")

uploaded_file = st.file_uploader(
    "Upload Garbage Image", 
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # ===============================
    # PREPROCESS
    # ===============================
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # ===============================
    # PREDICT
    # ===============================
    predictions = model.predict(img_array)

    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)

    st.subheader("Prediction")
    st.success(predicted_class)

    st.subheader("Confidence")
    st.write(f"{confidence:.2f}")

    # ===============================
    # TOP 3
    # ===============================
    st.subheader("Top 3 Predictions")
    top3 = np.argsort(predictions[0])[-3:][::-1]

    for i in top3:
        st.write(f"{class_names[i]} → {predictions[0][i]:.2f}")