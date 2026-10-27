"""
CreditLens-AI environment preflight.

Prints only whether things are PRESENT. It never prints, logs or
reads secret VALUES, and it does not open your .env file contents
beyond what python-dotenv loads into the process (values are not shown).

    python scripts/check_environment.py
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except Exception:
    pass

bad = 0


def line(ok: bool, text: str) -> None:
    global bad
    print(("OK    " if ok else "MISSING  ") + text)
    bad += 0 if ok else 1


print("Python packages")
for mod in ("fastapi", "uvicorn", "sqlalchemy", "asyncpg", "dotenv",
            "google.genai", "shap", "lightgbm", "xgboost", "catboost",
            "sklearn", "pandas", "pyarrow", "joblib", "streamlit",
            "requests", "reportlab"):
    try:
        m = importlib.import_module(mod)
        line(True, f"{mod} {getattr(m, '__version__', '')}".strip())
    except Exception:
        line(False, f"{mod}  (pip install it)")

print("\nEnvironment variables (names only)")
for name in ("DATABASE_URL", "GEMINI_API_KEY"):
    line(bool(os.getenv(name)), f"{name} is set")

print("\nModel files")
for f in ("regularized_lightgbm.pkl", "optimized_xgboost.pkl",
          "optimized_catboost.pkl", "isotonic_calibrator.pkl",
          "feature_columns.json", "ensemble_config.json"):
    line((ROOT / "models" / f).exists(), f"models/{f}")

print("\nOptional (customer lookup by ID)")
p = ROOT / "data" / "processed" / "test_processed.parquet"
print(("OK    " if p.exists() else "NOTE  ")
      + "data/processed/test_processed.parquet"
      + ("" if p.exists() else "  (needed only for lookup by customer ID)"))

print("\nResult:", "ready" if not bad else f"{bad} item(s) need attention")
sys.exit(1 if bad else 0)
