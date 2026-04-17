# db/crud.py
"""
All database read/write helpers.
Each function takes an AsyncSession and returns plain Python objects or None.
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Optional

from sqlalchemy import select, desc, update
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import CageReading, Alert
from core.schemas import DetectionResult, CageStatus, LevelReading, AlertEvent
from core.config import STALE_MINUTES


# ══════════════════════════════════════════════════════════════════════════════
# WRITES
# ══════════════════════════════════════════════════════════════════════════════

async def save_detection(db: AsyncSession, result: DetectionResult) -> CageReading:
    """Persist a DetectionResult → cage_readings row. Returns the saved ORM row."""
    row = CageReading(
        cage_id      = result.cage_id,
        recorded_at  = result.timestamp,
        mouse_count  = result.mouse_count,
        water_pct    = result.water.pct,
        water_status = result.water.status,
        food_pct     = result.food.pct,
        food_status  = result.food.status,
        inference_ms = result.inference_ms,
        image_path   = result.image_path,
    )
    db.add(row)
    await db.commit()
    await db.refresh(row)
    return row


async def create_alert(
    db: AsyncSession,
    cage_id: str,
    alert_type: str,
) -> Alert:
    """Insert a new unresolved alert row."""
    alert = Alert(
        cage_id      = cage_id,
        alert_type   = alert_type,
        triggered_at = datetime.now(tz=timezone.utc),
        resolved_at  = None,
        notified     = False,
    )
    db.add(alert)
    await db.commit()
    await db.refresh(alert)
    return alert


async def resolve_alert(db: AsyncSession, alert_id: int) -> Optional[Alert]:
    """Mark an alert as resolved. Returns the updated row or None if not found."""
    stmt = (
        update(Alert)
        .where(Alert.id == alert_id, Alert.resolved_at.is_(None))
        .values(resolved_at=datetime.now(tz=timezone.utc))
        .returning(Alert)
    )
    result = await db.execute(stmt)
    await db.commit()
    return result.scalars().first()


# ══════════════════════════════════════════════════════════════════════════════
# READS — cage readings
# ══════════════════════════════════════════════════════════════════════════════

async def get_latest_reading(
    db: AsyncSession,
    cage_id: str,
) -> Optional[CageReading]:
    """Return the most recent cage_readings row for a given cage_id."""
    stmt = (
        select(CageReading)
        .where(CageReading.cage_id == cage_id)
        .order_by(desc(CageReading.recorded_at))
        .limit(1)
    )
    result = await db.execute(stmt)
    return result.scalars().first()


async def get_cage_status(
    db: AsyncSession,
    cage_id: str,
) -> Optional[CageStatus]:
    """
    Build a CageStatus from the latest reading.
    Sets is_stale=True if the last reading is older than STALE_MINUTES.
    """
    row = await get_latest_reading(db, cage_id)
    if row is None:
        return None

    stale_cutoff = datetime.now(tz=timezone.utc) - timedelta(minutes=STALE_MINUTES)
    recorded_at  = row.recorded_at

    # Make recorded_at timezone-aware if stored naively
    if recorded_at.tzinfo is None:
        recorded_at = recorded_at.replace(tzinfo=timezone.utc)

    return CageStatus(
        cage_id      = row.cage_id,
        last_updated = recorded_at,
        water        = LevelReading(pct=float(row.water_pct), status=row.water_status),
        food         = LevelReading(pct=float(row.food_pct),  status=row.food_status),
        mouse_count  = row.mouse_count or 0,
        is_stale     = recorded_at < stale_cutoff,
    )


async def get_all_cage_ids(db: AsyncSession) -> list[str]:
    """Return distinct cage_ids that have at least one reading."""
    stmt = select(CageReading.cage_id).distinct()
    result = await db.execute(stmt)
    return list(result.scalars().all())


async def get_readings_history(
    db: AsyncSession,
    cage_id: str,
    limit: int = 50,
) -> list[CageReading]:
    """Return the N most recent readings for a cage, newest first."""
    stmt = (
        select(CageReading)
        .where(CageReading.cage_id == cage_id)
        .order_by(desc(CageReading.recorded_at))
        .limit(limit)
    )
    result = await db.execute(stmt)
    return list(result.scalars().all())


# ══════════════════════════════════════════════════════════════════════════════
# READS — alerts
# ══════════════════════════════════════════════════════════════════════════════

async def get_open_alerts(
    db: AsyncSession,
    cage_id: Optional[str] = None,
) -> list[Alert]:
    """Return all unresolved alerts, optionally filtered by cage_id."""
    stmt = select(Alert).where(Alert.resolved_at.is_(None))
    if cage_id:
        stmt = stmt.where(Alert.cage_id == cage_id)
    stmt = stmt.order_by(desc(Alert.triggered_at))
    result = await db.execute(stmt)
    return list(result.scalars().all())


async def get_alert_by_id(db: AsyncSession, alert_id: int) -> Optional[Alert]:
    result = await db.execute(select(Alert).where(Alert.id == alert_id))
    return result.scalars().first()