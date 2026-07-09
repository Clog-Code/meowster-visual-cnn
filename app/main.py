import os
import tempfile
import cv2
from collections import Counter
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.model import predict_emotion, get_model
from app.schemas import PredictionResponse, HealthResponse, VideoPredictionResponse

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
    
@app.post("/predict_video", response_model=VideoPredictionResponse)
async def predict_video(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1] if file.filename else ".mp4"
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_video:
        temp_video.write(await file.read())
        temp_video_path = temp_video.name

    predictions = []

    try:
        cap = cv2.VideoCapture(temp_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        frame_interval = max(1, int(fps)) if fps > 0 else 30
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break # Video is over

            if frame_count % frame_interval == 0:
                success, encoded_image = cv2.imencode('.jpg', frame)
                if success:
                    image_bytes = encoded_image.tobytes()
                    
                    result = predict_emotion(image_bytes)
                    predictions.append(result["predicted_emotion"])

            frame_count += 1

        cap.release()
    finally:
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)

    if not predictions:
        raise HTTPException(status_code=400, detail="Could not extract any frames from the video")

    emotion_counts = Counter(predictions)
    overall_emotion = emotion_counts.most_common(1)[0][0]

    return {
        "overall_emotion": overall_emotion,
        "frame_breakdown": dict(emotion_counts),
        "total_frames_analyzed": len(predictions)
    }
