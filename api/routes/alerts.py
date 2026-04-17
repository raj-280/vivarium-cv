# api/routes/alerts.py
"""
GET  /alerts                  — list all open alerts (optionally filter by cage_id)
GET  /alerts/{alert_id}       — get a single alert
POST /alerts/{alert_id}/resolve — mark alert as resolved
POST /alerts/trigger          — manually trigger an alert (useful for testing)
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from db.session import get_db
from db.crud import get_open_alerts, get_alert_by_id, resolve_alert, create_alert
from core.schemas import AlertEvent

logger = logging.getLogger("vivarium.alerts")
router = APIRouter(prefix="/alerts", tags=["alerts"])


# ── Response model ────────────────────────────────────────────────────────────

class AlertResponse(BaseModel):
    id:           int
    cage_id:      str
    alert_type:   str
    triggered_at: object   # datetime
    resolved_at:  Optional[object] = None
    notified:     bool


class TriggerAlertRequest(BaseModel):
    cage_id:    str
    alert_type: str   # WATER_LOW | WATER_CRITICAL | FOOD_LOW | FOOD_CRITICAL


VALID_ALERT_TYPES = {"WATER_LOW", "WATER_CRITICAL", "FOOD_LOW", "FOOD_CRITICAL"}


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/", response_model=list[AlertResponse])
async def list_alerts(
    cage_id: Optional[str] = Query(default=None, description="Filter by cage_id"),
    db: AsyncSession = Depends(get_db),
):
    """Return all unresolved alerts, newest first. Optionally filter by cage."""
    alerts = await get_open_alerts(db, cage_id=cage_id)
    return [_to_response(a) for a in alerts]


@router.get("/{alert_id}", response_model=AlertResponse)
async def get_alert(
    alert_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Return a single alert by ID."""
    alert = await get_alert_by_id(db, alert_id)
    if alert is None:
        raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found.")
    return _to_response(alert)


@router.post("/{alert_id}/resolve", response_model=AlertResponse)
async def resolve(
    alert_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Mark an alert as resolved (sets resolved_at to now)."""
    alert = await resolve_alert(db, alert_id)
    if alert is None:
        raise HTTPException(
            status_code=404,
            detail=f"Alert {alert_id} not found or already resolved.",
        )
    return _to_response(alert)


@router.post("/trigger", response_model=AlertResponse, status_code=201)
async def trigger_alert(
    body: TriggerAlertRequest,
    db:   AsyncSession = Depends(get_db),
):
    """
    Manually create an alert. Useful for Swagger testing and integration tests.
    Also fires the webhook if ALERT_WEBHOOK_URL is set.
    """
    if body.alert_type not in VALID_ALERT_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid alert_type '{body.alert_type}'. "
                   f"Valid: {sorted(VALID_ALERT_TYPES)}",
        )

    alert = await create_alert(db, cage_id=body.cage_id, alert_type=body.alert_type)

    # Fire webhook asynchronously (don't fail the request if it errors)
    await _fire_webhook(body.cage_id, body.alert_type)

    return _to_response(alert)


# ── Webhook helper ────────────────────────────────────────────────────────────

async def _fire_webhook(cage_id: str, alert_type: str) -> None:
    """POST alert payload to ALERT_WEBHOOK_URL if configured."""
    url = os.getenv("ALERT_WEBHOOK_URL", "")
    if not url:
        return
    payload = {"cage_id": cage_id, "alert_type": alert_type}
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            logger.info("Webhook fired for %s/%s → %d", cage_id, alert_type, resp.status_code)
    except Exception as exc:
        logger.warning("Webhook failed for %s/%s: %s", cage_id, alert_type, exc)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_response(alert) -> AlertResponse:
    return AlertResponse(
        id           = alert.id,
        cage_id      = alert.cage_id,
        alert_type   = alert.alert_type,
        triggered_at = alert.triggered_at,
        resolved_at  = alert.resolved_at,
        notified     = alert.notified,
    )