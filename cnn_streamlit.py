"""
Fashion MNIST — Standalone Streamlit App (No FastAPI needed)
=============================================================
Works locally AND on Streamlit Cloud.
Run: streamlit run cnn_streamlit.py
"""

import io
import numpy as np
import streamlit as st
from PIL import Image
import plotly.graph_objects as go

# ─────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fashion MNIST Classifier",
    page_icon="👗",
    layout="centered"
)

# ─────────────────────────────────────────────────────────────
# Label map
# ─────────────────────────────────────────────────────────────
LABEL_NAMES = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot"
}

# ─────────────────────────────────────────────────────────────
# Load model — cached so it only loads once
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    import tensorflow as tf
    try:
        model = tf.keras.models.load_model("fashionmnist.keras")
        return model, None
    except Exception as e:
        return None, str(e)

# ─────────────────────────────────────────────────────────────
# Preprocess image
# ─────────────────────────────────────────────────────────────
def preprocess(file_bytes: bytes) -> np.ndarray:
    img = (Image.open(io.BytesIO(file_bytes))
                 .convert("L")        # grayscale
                 .resize((28, 28)))   # 28x28
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr.reshape(1, 28, 28, 1)  # (1, 28, 28, 1)

# ─────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────
st.title("👗 Fashion MNIST Classifier")
st.markdown("Upload a clothing image and the AI will predict what it is.")
st.divider()

# Load model
model, error = load_model()

if error:
    st.error(f"❌ Could not load model: {error}")
    st.info("Make sure `fashionmnist.keras` is in the same folder as this script.")
    st.stop()
else:
    st.success("✅ Model loaded successfully!")

st.divider()

# ─────────────────────────────────────────────────────────────
# Upload
# ─────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload a clothing image (JPG / PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    file_bytes = uploaded_file.read()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Uploaded Image")
        img = Image.open(io.BytesIO(file_bytes))
        st.image(img, use_container_width=True)

    with col2:
        st.subheader("🤖 Prediction")
        with st.spinner("Classifying..."):
            try:
                arr         = preprocess(file_bytes)
                preds       = model.predict(arr, verbose=0)[0]   # shape (10,)
                pred_idx    = int(np.argmax(preds))
                pred_label  = LABEL_NAMES[pred_idx]
                confidence  = float(preds[pred_idx]) * 100

                st.metric("Predicted Class", pred_label)
                st.metric("Confidence", f"{confidence:.2f}%")

            except Exception as e:
                st.error(f"Prediction failed: {e}")
                preds = None

    # ── Probability bar chart ─────────────────────────────────
    if preds is not None:
        st.divider()
        st.subheader("📊 All Class Probabilities")

        labels = list(LABEL_NAMES.values())
        values = [float(p) * 100 for p in preds]
        colors = [
            "#2ecc71" if i == pred_idx else "#3498db"
            for i in range(10)
        ]

        fig = go.Figure(go.Bar(
            x=labels,
            y=values,
            marker_color=colors,
            text=[f"{v:.1f}%" for v in values],
            textposition="outside"
        ))
        fig.update_layout(
            yaxis_title="Probability (%)",
            xaxis_title="Class",
            yaxis=dict(range=[0, max(values) * 1.25]),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=420,
            margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Raw probabilities
        with st.expander("🔍 Raw Probabilities"):
            for i, (lbl, prob) in enumerate(zip(labels, values)):
                st.write(f"**{lbl}**: {prob:.2f}%")

st.divider()
st.caption("Powered by TensorFlow + Streamlit")
st.caption("Powered by FastAPI + TensorFlow + Streamlit")
