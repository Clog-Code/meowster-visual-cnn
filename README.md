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
├── requirements.txt
├── run.sh
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run the server

```bash
bash run.sh
# or directly:
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
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
curl -X POST -F "file=@test_images/sample_cat.jpg" http://localhost:8000/predict
```

Example response:

```json
{
  "predicted_emotion": "Happy",
  "confidence": 0.87,
  "all_probabilities": {
    "Angry": 0.02,
    "Happy": 0.87,
    "Sad": 0.06,
    "Relaxed": 0.05
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
pytest tests/
```

Put a sample image at `test_images/sample_cat.jpg` first so the prediction
test doesn't get skipped.

## Deploying

A `Dockerfile` is included for containerized deployment (e.g. to Cloud Run,
Fly.io, Render, etc.):

```bash
docker build -t cat-emotion-api .
docker run -p 8000:8000 cat-emotion-api
```

## Flutter integration

From your Flutter app, send a `multipart/form-data` POST request to
`http://<your-server>:8000/predict` with the image file under the field name
`file`, then parse the JSON response shown above.
