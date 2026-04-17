# db/session.py
"""
Async SQLAlchemy engine + session factory.
FastAPI routes get a session via the get_db() dependency.
"""
import os
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from db.models import Base

# ── Engine ────────────────────────────────────────────────────────────────────
DB_URL = os.getenv(
    "DB_URL",
    "postgresql+asyncpg://postgres:password@localhost:5432/vivarium",
)

engine = create_async_engine(
    DB_URL,
    echo=False,          # set True to log every SQL statement
    pool_pre_ping=True,  # verify connection health before use
    pool_size=5,
    max_overflow=10,
)

# ── Session factory ───────────────────────────────────────────────────────────
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,   # keep attrs accessible after commit
    autoflush=False,
    autocommit=False,
)

# ── FastAPI dependency ────────────────────────────────────────────────────────
async def get_db() -> AsyncSession:
    """
    Yields an AsyncSession per request and guarantees close on exit.
    Use as: db: AsyncSession = Depends(get_db)
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise

# ── Table creation helper (called from lifespan) ──────────────────────────────
async def create_tables() -> None:
    """Create all tables if they don't exist yet. Safe to call on every startup."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)