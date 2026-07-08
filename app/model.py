import os
os.environ.setdefault("KERAS_BACKEND", "jax")

import io
import numpy as np
from PIL import Image
import keras

# Hugging Face model repo id
MODEL_ID = "hf://Belall87/Cat-Emotion-Classification-with-CNN"

# TODO: Verify these against the actual model card / model.summary() output
INPUT_SIZE = (224, 224)
CLASS_NAMES = ["angry", "normal", "rested", "sad","surprised"]

_model = None  # lazy-loaded, cached singleton


def get_model():
    """Load the model once and cache it. Subsequent calls return the cached instance."""
    global _model
    if _model is None:
        print("Loading model from Hugging Face...")
        _model = keras.saving.load_model(MODEL_ID)
        print("Model loaded.")
        print("Input shape:", _model.input_shape)
        print("Output shape:", _model.output_shape)
    return _model


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Convert raw image bytes into a normalized numpy array ready for prediction."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(INPUT_SIZE)
    arr = np.array(image, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)  # add batch dimension
    return arr


def predict_emotion(image_bytes: bytes) -> dict:
    """Run the full pipeline: preprocess -> predict -> format response."""
    model = get_model()
    input_array = preprocess_image(image_bytes)
    probs = model.predict(input_array)[0]

    predicted_index = int(np.argmax(probs))
    predicted_label = CLASS_NAMES[predicted_index]
    confidence = float(probs[predicted_index])

    return {
        "predicted_emotion": predicted_label,
        "confidence": confidence,
        "all_probabilities": {
            CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))
        },
    }
