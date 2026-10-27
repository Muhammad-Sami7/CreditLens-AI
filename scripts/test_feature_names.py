"""Offline tests for plain-language feature names (display only)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "frontend"))

from components.feature_names import friendly_name, replace_feature_names  # noqa: E402

failed = 0


def check(name, ok):
    global failed
    print(("PASS  " if ok else "FAIL  ") + name)
    failed += 0 if ok else 1


check("documented name", friendly_name("EXT_SOURCE_3") == "External credit score 3")
check("engineered name", friendly_name("CREDIT_TO_INCOME_RATIO") == "Loan amount compared with income")
check("document flag", friendly_name("FLAG_DOCUMENT_3") == "Document 3 provided")
check("missing flag", friendly_name("EXT_SOURCE_1_MISSING") == "External credit score 1 - not recorded")
check("history prefix", friendly_name("INST_AVG_DELAY").startswith("Instalment payments: average"))
check("category value untouched", friendly_name("Cash loans") == "Cash loans")
check("free text swap",
      replace_feature_names("High AMT_ANNUITY and low EXT_SOURCE_2.")
      == "High Regular loan payment and low External credit score 2.")
check("text without names unchanged", replace_feature_names("Low risk.") == "Low risk.")
check("empty text", replace_feature_names("") == "")

# every production feature gets a non-technical label (no raw ALL_CAPS_WITH_UNDERSCORES left)
path = ROOT / "models" / "feature_columns.json"
if path.exists():
    cols = json.loads(path.read_text())
    leftover = [c for c in cols if "_" in friendly_name(c) and friendly_name(c).isupper()]
    check("no raw technical name is shown as-is", not leftover)

print("\nResult:", "all checks passed" if not failed else f"{failed} FAILED")
sys.exit(1 if failed else 0)
