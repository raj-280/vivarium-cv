# api/main.py
"""
FastAPI application entry point.

Lifespan:
  - Creates DB tables on startup (idempotent — safe to run every time)
  - Logs which backend is active

Routes:
  POST /infer             — upload a frame, get DetectionResult
  GET  /levels/{cage_id} — latest water + food readings
  GET  /levels/{cage_id}/history
  GET  /cages/            — all cages status
  GET  /cages/{cage_id}  — single cage status
  GET  /alerts/          — open alerts
  POST /alerts/trigger   — create an alert manually (testing)
  POST /alerts/{id}/resolve
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.middleware import register_middleware
from api.routes import infer, levels, cages, alerts
from db.session import create_tables

logger = logging.getLogger("vivarium.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup → yield → shutdown."""
    # ── Startup ───────────────────────────────────────────────────
    logger.info("Starting Vivarium CV API …")
    await create_tables()
    logger.info("DB tables ready.")

    backend = os.getenv("BACKEND", "yolo").lower()
    logger.info("Active backend: %s", backend.upper())

    yield  # ← app runs here

    # ── Shutdown ──────────────────────────────────────────────────
    logger.info("Vivarium CV API shutting down.")


app = FastAPI(
    title       = "Vivarium CV API",
    version     = "0.2.0",
    description = (
        "Computer vision API for vivarium cage monitoring.\n\n"
        "Upload a cage frame via **POST /infer** to get mouse count + "
        "water/food level readings.\n\n"
        "Switch detection backend via `BACKEND=yolo` or `BACKEND=ssd` in `.env`."
    ),
    lifespan=lifespan,
)

register_middleware(app)

app.include_router(infer.router)
app.include_router(levels.router)
app.include_router(cages.router)
app.include_router(alerts.router)