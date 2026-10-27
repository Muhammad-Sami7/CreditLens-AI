"""
===========================================================
CreditLens-AI
Phase 2 Live API Test
===========================================================

Run against your REAL running API (models, Gemini, PostgreSQL):

    uvicorn src.main:app --reload        # in another terminal
    python scripts/test_api_phase2_live.py

Optional: set CREDITLENS_API_URL, e.g. http://127.0.0.1:8000/api/v1
"""

from __future__ import annotations

import os
import sys

import requests

BASE = os.getenv(
    "CREDITLENS_API_URL", "http://127.0.0.1:8000/api/v1"
).rstrip("/")

TIMEOUT = 180  # SHAP + Gemini can be slow

failures = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global failures
    print(("PASS  " if condition else "FAIL  ") + name
          + (f"  [{detail}]" if detail and not condition else ""))
    if not condition:
        failures += 1


def main() -> int:

    r = requests.get(f"{BASE}/health", timeout=10)
    check("health", r.ok and r.json().get("status") == "healthy")

    r = requests.get(f"{BASE}/customers/sample", params={"limit": 3},
                     timeout=TIMEOUT)
    check("sample customers", r.ok and r.json()["customer_ids"],
          r.text[:200])
    if not r.ok:
        return 1
    customer_id = r.json()["customer_ids"][0]
    print(f"      using customer {customer_id}")

    r = requests.post(f"{BASE}/assess/{customer_id}", timeout=TIMEOUT)
    check("assess by ID", r.ok, r.text[:300])
    if not r.ok:
        return 1
    body = r.json()

    check("original fields present",
          all(k in body for k in (
              "prediction", "risk_level", "decision", "probability",
              "ensemble_probability", "threshold",
              "model_probabilities", "llm_explanation")))
    check("prediction_id returned",
          isinstance(body.get("prediction_id"), int))
    shap = body.get("shap") or {}
    check("structured SHAP returned", bool(shap.get("top_factors")))
    check("feature_completeness returned",
          body.get("feature_completeness", {}).get("expected_count") == 297,
          str(body.get("feature_completeness")))

    check("null_count reported",
          "null_count" in body.get("feature_completeness", {}))

    # A customer ID that does not exist must be a real 404 (not a crash)
    # and must be distinguishable from an unreachable API.
    r = requests.post(f"{BASE}/assess/__no_such_customer__", timeout=30)
    check("unknown customer -> 404", r.status_code == 404)

    r = requests.get(f"{BASE}/customers/{customer_id}/predictions",
                     timeout=30)
    history = r.json().get("predictions", []) if r.ok else []
    check("history contains the new prediction",
          any(p["prediction_id"] == body["prediction_id"]
              for p in history))

    r = requests.get(f"{BASE}/customers/__nobody__/predictions", timeout=30)
    check("history 200 + empty for unknown customer",
          r.ok and r.json()["predictions"] == [])

    r = requests.get(f"{BASE}/dashboard/summary", timeout=30)
    summary = r.json() if r.ok else {}
    check("dashboard summary",
          r.ok and summary.get("total_assessments", 0) >= 1,
          r.text[:200])
    check("summary high + low == total",
          summary.get("high_risk_assessments", 0)
          + summary.get("low_risk_assessments", 0)
          == summary.get("total_assessments"))

    print("\nALL PASSED" if not failures else f"\n{failures} FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
