from pydantic import BaseModel
from typing import Dict


class PredictionResponse(BaseModel):
    predicted_emotion: str
    confidence: float
    all_probabilities: Dict[str, float]


class HealthResponse(BaseModel):
    status: str
    message: str
    
class VideoPredictionResponse(BaseModel):
    overall_emotion: str
    frame_breakdown: Dict[str, int]
    total_frames_analyzed: int
