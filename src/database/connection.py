"""
============================================================
CreditLens-AI
Database Connection
============================================================

Provides the asynchronous SQLAlchemy engine,
session factory, and database session dependency.
"""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator

from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase


# ------------------------------------------------------------
# Environment
# ------------------------------------------------------------

load_dotenv()


DATABASE_URL = os.getenv("DATABASE_URL")


if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL environment variable is not configured."
    )


# ------------------------------------------------------------
# Base
# ------------------------------------------------------------

class Base(DeclarativeBase):
    """
    Base class for all SQLAlchemy ORM models.
    """

    pass


# ------------------------------------------------------------
# Engine
# ------------------------------------------------------------

engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    pool_pre_ping=True,
)


# ------------------------------------------------------------
# Session Factory
# ------------------------------------------------------------

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


# ------------------------------------------------------------
# Database Session
# ------------------------------------------------------------

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Provide an asynchronous database session.

    The session is automatically closed after
    the request finishes.
    """

    async with AsyncSessionLocal() as session:

        try:

            yield session

        except Exception:

            await session.rollback()

            raise


# ------------------------------------------------------------
# Database Shutdown
# ------------------------------------------------------------

async def close_database() -> None:
    """
    Dispose of the SQLAlchemy engine.
    """

    await engine.dispose()