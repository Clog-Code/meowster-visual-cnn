from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.model import predict_emotion, get_model
from app.schemas import PredictionResponse, HealthResponse

app = FastAPI(
    title="Cat Emotion Classification API",
    description="Upload a cat photo and get back a predicted emotion.",
    version="1.0.0",
)

# Allow the Flutter app (web/mobile/desktop) to call this API.
# Restrict allow_origins to your actual app's origin(s) in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def load_model_on_startup():
    # Warm up / load the model once when the server starts,
    # so the first real request isn't slow.
    get_model()


@app.get("/", response_model=HealthResponse)
def health_check():
    return {"status": "ok", "message": "Cat Emotion Classification API is running"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        image_bytes = await file.read()
        result = predict_emotion(image_bytes)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
