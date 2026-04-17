# api/routes/levels.py
"""
GET /levels/{cage_id}         — latest water + food readings for a cage
GET /levels/{cage_id}/history — last N readings
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from db.session import get_db
from db.crud import get_latest_reading, get_readings_history
from core.schemas import LevelReading
from pydantic import BaseModel
from datetime import datetime
from typing import Optional

router = APIRouter(prefix="/levels", tags=["levels"])


class LevelResponse(BaseModel):
    cage_id:     str
    recorded_at: datetime
    water:       LevelReading
    food:        LevelReading
    inference_ms: Optional[int] = None


class HistoryItem(BaseModel):
    recorded_at:  datetime
    water:        LevelReading
    food:         LevelReading
    mouse_count:  int
    inference_ms: Optional[int] = None


@router.get("/{cage_id}", response_model=LevelResponse)
async def get_levels(
    cage_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Return the most recent water and food level readings for a cage."""
    row = await get_latest_reading(db, cage_id)
    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"No readings found for cage '{cage_id}'.",
        )
    return LevelResponse(
        cage_id     = row.cage_id,
        recorded_at = row.recorded_at,
        water       = LevelReading(pct=float(row.water_pct), status=row.water_status),
        food        = LevelReading(pct=float(row.food_pct),  status=row.food_status),
        inference_ms= row.inference_ms,
    )


@router.get("/{cage_id}/history", response_model=list[HistoryItem])
async def get_history(
    cage_id: str,
    limit:   int = Query(default=50, ge=1, le=500, description="Max readings to return"),
    db: AsyncSession = Depends(get_db),
):
    """Return the last N readings for a cage, newest first."""
    rows = await get_readings_history(db, cage_id, limit=limit)
    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"No history found for cage '{cage_id}'.",
        )
    return [
        HistoryItem(
            recorded_at = r.recorded_at,
            water       = LevelReading(pct=float(r.water_pct), status=r.water_status),
            food        = LevelReading(pct=float(r.food_pct),  status=r.food_status),
            mouse_count = r.mouse_count or 0,
            inference_ms= r.inference_ms,
        )
        for r in rows
    ]