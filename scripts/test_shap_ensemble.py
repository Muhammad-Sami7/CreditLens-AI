"""
Checks that the ensemble SHAP explanation is faithful (needs the real
models in models/ and the processed data; no database or Gemini needed).

For each model: base value + sum(SHAP) must equal that model's raw score.
For the ensemble: contributions must equal the weighted sum of the three.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.explainability.shap_explainer import SHAPExplainer  # noqa: E402

ex = SHAPExplainer()

df = pd.read_parquet(ROOT / "data" / "processed" / "test_processed.parquet")
rows = df[ex.feature_columns].head(5)

failures = 0


def check(name, ok, detail=""):
    global failures
    print(("PASS  " if ok else "FAIL  ") + name + (f"  [{detail}]" if not ok else ""))
    failures += 0 if ok else 1


def raw_margin(name, model, frame):
    if name == "lightgbm":
        return model.predict(frame, raw_score=True)
    if name == "xgboost":
        return model.predict(frame, output_margin=True)
    return model.predict(frame, prediction_type="RawFormulaVal")


for i in range(len(rows)):
    one = rows.iloc[[i]]
    result = ex.explain(one, top_n=5)

    for name, model in ex.models.items():
        phi = ex._calculate_shap_values(ex._prepare_features(one), name)[0]
        base = ex._base_value(ex.explainers[name])
        gap = abs(base + phi.sum() - float(raw_margin(name, model, one)[0]))
        check(f"row {i} {name}: base + sum(SHAP) == model score", gap < 1e-3, f"gap={gap:.2e}")

    weighted = sum(
        ex.weights[n] * ex._calculate_shap_values(ex._prepare_features(one), n)[0]
        for n in ex.models
    )
    combined = np.array(result["shap_values"])
    check(f"row {i}: ensemble = weighted sum of the 3 models",
          np.allclose(weighted, combined, atol=1e-9))
    check(f"row {i}: all three models explained",
          result["models_explained"] == ["lightgbm", "xgboost", "catboost"])
    top = result["top_factors"][0]
    check(f"row {i}: top factor reports model agreement",
          0 <= top["models_agreeing"] <= 3 and set(top["model_shap_values"]) == set(ex.models))

print("\nResult:", "all checks passed" if not failures else f"{failures} FAILED")
sys.exit(1 if failures else 0)
