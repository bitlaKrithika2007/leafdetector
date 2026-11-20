import json
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf

st.set_page_config(page_title="Model Classifier", page_icon="🧠", layout="centered")
st.title("Image Classifier")
st.write("Upload an image to run inference with `ldm.keras`.")


@st.cache_resource
def load_model():
    return tf.keras.models.load_model("ldm.keras")


@st.cache_data
def load_class_names():
    with open("classnames.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    # Supports either a list ["cat", "dog"] or dict {"0": "cat", "1": "dog"}
    if isinstance(data, dict):
        return [data[str(i)] for i in range(len(data))]
    return list(data)


def preprocess_image(image: Image.Image, target_size: tuple[int, int]) -> np.ndarray:
    img = image.convert("RGB").resize(target_size)
    arr = np.asarray(img, dtype=np.float32)
    # Adjust scaling if your model expects [-1, 1] or unscaled [0, 255]
    arr = arr / 255.0
    return np.expand_dims(arr, axis=0)


try:
    model = load_model()
    class_names = load_class_names()
except Exception as e:
    st.error(f"Failed to load model or classnames.json: {e}")
    st.stop()

# Infer input height/width from model input_shape (e.g., (None, 224, 224, 3))
input_shape = model.input_shape
if isinstance(input_shape, list):
    input_shape = input_shape[0]
target_size = (int(input_shape[1]), int(input_shape[2]))

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict", type="primary"):
        with st.spinner("Running prediction..."):
            batch = preprocess_image(image, target_size)
            preds = model.predict(batch, verbose=0)
            probs = preds[0]

            # Apply softmax if model outputs raw logits (values outside [0, 1] or sum != 1)
            if np.min(probs) < 0.0 or np.max(probs) > 1.0 or not np.isclose(np.sum(probs), 1.0, atol=1e-2):
                probs = tf.nn.softmax(probs).numpy()

            top_idx = int(np.argmax(probs))
            top_label = class_names[top_idx] if top_idx < len(class_names) else f"Class {top_idx}"
            top_conf = float(probs[top_idx])

            st.subheader(f"Prediction: {top_label}")
            st.write(f"Confidence: **{top_conf * 100:.2f}%**")

            st.write("### Class Probabilities")
            prob_dict = {
                (class_names[i] if i < len(class_names) else f"Class {i}"): float(probs[i])
                for i in range(len(probs))
            }
            st.bar_chart(prob_dict)