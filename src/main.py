"""
============================================================
CreditLens-AI
FastAPI Application
============================================================

Author: Muhammad Sami
"""

from __future__ import annotations

from fastapi import FastAPI

from src.api.routes import router


# ------------------------------------------------------------
# Application
# ------------------------------------------------------------

app = FastAPI(
    title="CreditLens-AI",
    description=(
        "AI Financial Risk Intelligence Platform "
        "for credit-risk assessment."
    ),
    version="1.0.0",
)


# ------------------------------------------------------------
# Routes
# ------------------------------------------------------------

app.include_router(router)


# ------------------------------------------------------------
# Root
# ------------------------------------------------------------

@app.get("/")
def root() -> dict[str, str]:

    return {
        "application": "CreditLens-AI",
        "version": "1.0.0",
        "status": "running",
    }