"""
Fashion MNIST — Streamlit UI
=============================
Make sure FastAPI is running first:
    uvicorn fashion_api:app --reload

Then run this:
    streamlit run streamlit_app.py
"""

import streamlit as st
import requests
from PIL import Image
import io
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
# FastAPI URL
# ─────────────────────────────────────────────────────────────
API_URL = "http://127.0.0.1:8000"

# ─────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────
st.title("👗 Fashion MNIST Classifier")
st.markdown("Upload a clothing image and the AI will predict what it is.")
st.divider()

# ─────────────────────────────────────────────────────────────
# Check if API is running
# ─────────────────────────────────────────────────────────────
try:
    health = requests.get(f"{API_URL}/", timeout=3)
    if health.status_code == 200 and health.json().get("model_loaded"):
        st.success("✅ FastAPI server is running and model is loaded!")
    else:
        st.error("⚠️ Server is up but model failed to load.")
except Exception:
    st.error("❌ FastAPI server is not running. Start it with: `uvicorn fashion_api:app --reload`")
    st.stop()

st.divider()

# ─────────────────────────────────────────────────────────────
# Image Upload
# ─────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload a clothing image (JPG / PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    col1, col2 = st.columns(2)

    # Show uploaded image
    with col1:
        st.subheader("📷 Uploaded Image")
        img = Image.open(uploaded_file)
        st.image(img, use_container_width=True)

    # Send to FastAPI and show result
    with col2:
        st.subheader("🤖 Prediction")
        with st.spinner("Classifying..."):
            try:
                uploaded_file.seek(0)
                response = requests.post(
                    f"{API_URL}/predict",
                    files={"file": (uploaded_file.name,
                                    uploaded_file.read(),
                                    uploaded_file.type)},
                    timeout=10
                )

                if response.status_code == 200:
                    result = response.json()

                    # Main prediction
                    st.metric(
                        label="Predicted Class",
                        value=result["predicted_label"],
                    )
                    st.metric(
                        label="Confidence",
                        value=f"{result['confidence']}%"
                    )

                else:
                    st.error(f"API Error {response.status_code}: {response.json().get('detail')}")
                    result = None

            except Exception as e:
                st.error(f"Request failed: {e}")
                result = None

    # ── Probability bar chart ─────────────────────────────────
    if uploaded_file and result:
        st.divider()
        st.subheader("📊 All Class Probabilities")

        probs  = result["all_probabilities"]
        labels = list(probs.keys())
        values = list(probs.values())

        # Highlight the predicted bar
        colors = [
            "#2ecc71" if lbl == result["predicted_label"] else "#3498db"
            for lbl in labels
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
            yaxis=dict(range=[0, max(values) * 1.2]),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=400,
            margin=dict(t=20, b=20),
        )

        st.plotly_chart(fig, use_container_width=True)

        # ── Raw JSON result ───────────────────────────────────
        with st.expander("🔍 Raw API Response"):
            st.json(result)

# ─────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────
st.divider()
st.caption("Powered by FastAPI + TensorFlow + Streamlit")