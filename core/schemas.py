from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class DetectionResult(BaseModel):
    cage_id: str
    timestamp: datetime
    mouse_count: int
    water_level_pct: float
    water_status: str
    food_level_pct: float
    food_status: str
    inference_ms: Optional[int] = None

class LevelReading(BaseModel):
    pct: float
    status: str  # OK / LOW / CRITICAL

class CageStatus(BaseModel):
    cage_id: str
    last_updated: datetime
    water: LevelReading
    food: LevelReading
    mouse_count: int