# Cat Emotion Classification API

A FastAPI wrapper around the [Belall87/Cat-Emotion-Classification-with-CNN](https://huggingface.co/Belall87/Cat-Emotion-Classification-with-CNN)
Hugging Face model, so it can be called from a Flutter (or any other) app over HTTP.

## Project structure

```
cat-emotion-api/
├── app/
│   ├── __init__.py
│   ├── main.py         # FastAPI app + routes
│   ├── model.py        # Model loading + preprocessing + prediction logic
│   └── schemas.py       # Pydantic response models
├── tests/
│   ├── __init__.py
│   └── test_predict.py
├── test_images/
│   └── README.md         # put a sample_cat.jpg here for local testing
├── .env.example
├── .gitignore
├── Dockerfile
├── pyproject.toml
├── run.sh
└── README.md
```

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

1. Install `uv` (if you don't have it):

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

2. Install dependencies (uv creates and manages the virtual environment automatically):

```bash
uv sync
source .venv/bin/activat
```

## Run the server

```bash
bash run.sh
# or directly:
uv run uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload # Reserve 8000 for amd-pet-agentic
```

The first request (or startup, since the model loads on startup) will take a
little while as it downloads the model from Hugging Face.

## Endpoints

### `GET /`

Health check.

```json
{ "status": "ok", "message": "Cat Emotion Classification API is running" }
```

### `POST /predict`

Upload an image file (`multipart/form-data`, field name `file`) and get back
the predicted emotion.

```bash
curl -X POST -F "file=@test_images/sample_cat.jpg" http://localhost:8001/predict
```

Example response:

```json
{
  "predicted_emotion": "sad",
  "confidence": 0.7967,
  "is_video": false,
  "detail_breakdown": {
    "angry": 0.0382,
    "normal": 0.0688,
    "rested": 0.0289,
    "sad": 0.7967,
    "surprised": 0.0674
  }
}
```

### `POST /predict_video`

Upload a video file (`multipart/form-data`, field name `file`). The API processes the video by sampling frames at 1 frame per second and returns an aggregated emotion evaluation using the same unified response schema.

```bash
curl -X POST -F "file=@test_images/cat_video.mp4;type=video/mp4" http://localhost:8001/predict_video
```

Example response:

```json
{
"predicted_emotion": "normal",
  "confidence": 0.7143,
  "is_video": true,
  "detail_breakdown": {
    "rested": 0.1429,
    "normal": 0.7143,
    "surprised": 0.1429
  }
}
```

## ⚠️ Before this works correctly, verify:

1. **`INPUT_SIZE` in `app/model.py`** — must match the exact image size the
   model expects. Check the Hugging Face model card, or run:

   ```python
   from app.model import get_model
   model = get_model()
   print(model.input_shape)
   ```

2. **`CLASS_NAMES` in `app/model.py`** — must match the actual label order
   the model was trained with. Check the model card / repo files for a
   `labels.txt`, `config.json`, or similar.
3. **Pixel normalization** — this code assumes 0–1 scaling
   (`pixel / 255.0`). Some models expect -1 to 1, or a specific
   `preprocess_input` function instead — check the model card.

## Testing

```bash
uv run pytest tests/
```

Put a sample image at `test_images/sample_cat.jpg` first so the prediction
test doesn't get skipped.

## Deploying

A `Dockerfile` is included for containerized deployment (e.g. to Cloud Run,
Fly.io, Render, etc.):

```bash
docker build -t cat-emotion-api .
docker run -p 8001:8001 cat-emotion-api
```

## Flutter integration

From your Flutter app, send a `multipart/form-data` POST request to
`http://<your-server>:8001/predict` with the image file under the field name
`file`, then parse the JSON response shown above.
