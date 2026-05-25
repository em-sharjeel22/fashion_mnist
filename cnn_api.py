"""
Fashion MNIST — FastAPI Inference Server
========================================
Run  : uvicorn fashion_api:app --reload
Docs : http://127.0.0.1:8000/docs
"""

import io
import numpy as np
import tensorflow as tf
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

# ─────────────────────────────────────────────────────────────
# App instance
# ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="Fashion MNIST Classifier",
    description="Upload a clothing image and get an emotion/class prediction.",
    version="1.0.0"
)

# ─────────────────────────────────────────────────────────────
# Label map — 10 Fashion MNIST classes
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
# Load model once at startup  (not on every request)
# ─────────────────────────────────────────────────────────────
MODEL_PATH = "fashionmnist.keras"

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"✅ Model loaded from '{MODEL_PATH}'")
except Exception as e:
    model = None
    print(f"❌ Could not load model: {e}")


# ─────────────────────────────────────────────────────────────
# Helper — preprocess an uploaded image
# ─────────────────────────────────────────────────────────────
def preprocess_image(file_bytes: bytes) -> np.ndarray:
    """
    1. Open image from raw bytes
    2. Convert to grayscale  (L mode)
    3. Resize to 28 × 28    (model input size)
    4. Normalise to [0, 1]
    5. Reshape to (1, 28, 28, 1)  for CNN
    """
    img = Image.open(io.BytesIO(file_bytes)).convert("L").resize((28, 28))
    arr = np.array(img, dtype=np.float32) / 255.0   # normalise
    arr = arr.reshape(1, 28, 28, 1)                  # add batch + channel dims
    return arr


# ─────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    """Health check — confirms the server is running."""
    return {
        "status": "running",
        "model_loaded": model is not None,
        "message": "POST an image to /predict to classify it."
    }


@app.get("/labels", tags=["Info"])
def get_labels():
    """Returns all 10 class labels."""
    return {"labels": LABEL_NAMES}


@app.post("/predict", tags=["Prediction"])
async def predict(file: UploadFile = File(...)):
    """
    Upload a clothing image (JPG / PNG) and receive:
    - predicted_class   : integer label (0-9)
    - predicted_label   : human-readable class name
    - confidence        : probability of the top prediction (%)
    - all_probabilities : probability for every class
    """

    # ── Guard: model must be loaded ──────────────────────────
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded. Make sure '{MODEL_PATH}' exists."
        )

    # ── Guard: only accept image files ───────────────────────
    if file.content_type not in ("image/jpeg", "image/png",
                                  "image/jpg", "image/webp"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file.content_type}'. "
                   "Please upload a JPG or PNG image."
        )

    # ── Read & preprocess ─────────────────────────────────────
    try:
        contents = await file.read()
        img_array = preprocess_image(contents)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Image processing failed: {e}")

    # ── Inference ─────────────────────────────────────────────
    try:
        predictions   = model.predict(img_array, verbose=0)   # shape (1, 10)
        probs         = predictions[0]                         # shape (10,)
        predicted_idx = int(np.argmax(probs))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    # ── Build response ────────────────────────────────────────
    all_probs = {
        LABEL_NAMES[i]: round(float(probs[i]) * 100, 2)
        for i in range(len(probs))
    }

    return JSONResponse(content={
        "predicted_class":   predicted_idx,
        "predicted_label":   LABEL_NAMES[predicted_idx],
        "confidence":        round(float(probs[predicted_idx]) * 100, 2),
        "all_probabilities": all_probs
    })