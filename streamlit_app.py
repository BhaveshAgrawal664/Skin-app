import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

st.title("Skin Disease Detector")

@st.cache_resource
def load_model():
    # Add compile=False to avoid batch_shape issue
    model = tf.keras.models.load_model(
        r"C:\Users\HP\Desktop\Skin-app\models\model_converted.keras",
        compile=False
    )
    return model

model = load_model()

uploaded = st.file_uploader("Upload a skin image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded image", use_column_width=True)

    # Preprocess image to 224x224 (change if your model input differs)
    img = img.resize((224, 224))
    x = np.array(img) / 255.0
    x = np.expand_dims(x, axis=0)

    preds = model.predict(x)
    st.write("Predictions:", preds)





