"""
Basic tests for the Cat Emotion Classification API.

Run with: pytest tests/

Note: These tests will actually load the model (they hit the real /predict
endpoint), so the first run may be slow while the model downloads from
Hugging Face. Make sure you have a test image at test_images/sample_cat.jpg.
"""

import os
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

SAMPLE_IMAGE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "test_images", "sample_cat.jpg"
)


def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_predict_rejects_non_image():
    response = client.post(
        "/predict",
        files={"file": ("test.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 400


def test_predict_with_sample_image():
    if not os.path.exists(SAMPLE_IMAGE_PATH):
        import pytest
        pytest.skip("No sample image found at test_images/sample_cat.jpg")

    with open(SAMPLE_IMAGE_PATH, "rb") as f:
        response = client.post(
            "/predict",
            files={"file": ("sample_cat.jpg", f, "image/jpeg")},
        )

    assert response.status_code == 200
    data = response.json()
    assert "predicted_emotion" in data
    assert "confidence" in data
    assert "all_probabilities" in data
