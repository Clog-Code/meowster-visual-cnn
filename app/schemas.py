from pydantic import BaseModel
from typing import Dict

class UnifiedPredictionResponse(BaseModel):
    predicted_emotion: str
    confidence: float
    is_video: bool
    detail_breakdown: Dict[str, float]

class HealthResponse(BaseModel):
    status: str
    message: str