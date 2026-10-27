"""Offline check of the PDF report (no API, database or Gemini needed)."""
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "frontend"))

from components.formatting import format_number  # noqa: E402
from components.report_pdf import build_assessment_pdf  # noqa: E402

sample = {
    "customer_id": "TEST-1", "prediction": 1, "risk_level": "High Risk",
    "decision": "Review Required", "probability": 0.25,
    "ensemble_probability": 0.4, "threshold": 0.18, "prediction_id": 1,
    "model_probabilities": {"lightgbm": .4, "xgboost": .4, "catboost": .4},
    "llm_explanation": {
        "summary": "Summary with \u201cquotes\u201d \u2013 and <tags> & symbols \u2265.",
        "risk_explanation": "Low EXT_SOURCE_3 mattered.", "key_risk_factors": ["A"],
        "protective_factors": ["B"], "analyst_actions": ["C"]},
    "shap": {"positive_factors": [], "negative_factors": []},
    "feature_completeness": None,
}

pdf = build_assessment_pdf(sample)
assert pdf.startswith(b"%PDF"), "not a PDF"
assert len(pdf) > 1500, "PDF suspiciously small"
sample["llm_explanation"] = None  # missing explanation must not crash
assert build_assessment_pdf(sample).startswith(b"%PDF")
print("PASS  PDF report builds (full and without explanation)")

# ---- readable numbers and plain-language names ----
assert format_number(49018.5) in ("49,018", "49,019")
assert format_number(1575000.0) == "1,575,000"
assert format_number(0.1062) == "0.1062"
assert format_number(560) == "560" and format_number(None) == "no data"
assert "e+" not in format_number(5e4) and "e+" not in format_number(1e4)

full = dict(sample)
full["llm_explanation"] = {"summary": "S", "risk_explanation": "Low EXT_SOURCE_3 mattered.",
                           "key_risk_factors": [], "protective_factors": [],
                           "analyst_actions": []}
full["shap"] = {
    "positive_factors": [
        {"rank": 1, "feature": "EXT_SOURCE_1", "feature_value": 0.0, "shap_value": 0.1078,
         "direction": "positive", "models_agreeing": 3},
        {"rank": 2, "feature": "AMT_ANNUITY", "feature_value": 49018.5, "shap_value": 0.1,
         "direction": "positive", "models_agreeing": 3}],
    "negative_factors": [
        {"rank": 1, "feature": "AMT_GOODS_PRICE", "feature_value": 1575000.0, "shap_value": -0.11,
         "direction": "negative", "models_agreeing": 3}],
}
full["feature_completeness"] = {"expected_count": 297, "provided_count": 290,
                                "missing_count": 7, "null_count": 0}
pdf = build_assessment_pdf(full)

with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as fh:
    fh.write(pdf)

try:
    text = subprocess.run(["pdftotext", "-layout", fh.name, "-"],
                          capture_output=True, text=True).stdout
except FileNotFoundError:
    text = None

if text:
    flat = " ".join(text.split())
    assert "1,575,000" in text and "49,0" in text, "numbers not readable"
    assert "e+0" not in text, "scientific notation leaked"
    assert "External credit score 1" in flat, "plain-language name missing"
    assert "EXT_SOURCE" not in flat, "technical name leaked"
    assert "7 of 297" in flat, "data note missing"
    print("PASS  PDF: readable numbers, plain-language names, data note")
else:
    print("SKIP  pdftotext not available")
