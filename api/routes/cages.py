# api/routes/cages.py
"""
GET /cages           — list all known cages with their current status
GET /cages/{cage_id} — status for a single cage
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from db.session import get_db
from db.crud import get_cage_status, get_all_cage_ids
from core.schemas import CageStatus

router = APIRouter(prefix="/cages", tags=["cages"])


@router.get("/", response_model=list[CageStatus])
async def list_cages(db: AsyncSession = Depends(get_db)):
    """
    Return current status for every cage that has at least one reading.
    Includes is_stale=True if the last reading is older than STALE_MINUTES.
    """
    cage_ids = await get_all_cage_ids(db)
    if not cage_ids:
        return []

    statuses = []
    for cage_id in cage_ids:
        status = await get_cage_status(db, cage_id)
        if status:
            statuses.append(status)

    return statuses


@router.get("/{cage_id}", response_model=CageStatus)
async def get_cage(
    cage_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Return current status for a single cage."""
    status = await get_cage_status(db, cage_id)
    if status is None:
        raise HTTPException(
            status_code=404,
            detail=f"Cage '{cage_id}' not found or has no readings yet.",
        )
    return status