"""
Threshold audit (read-only).

Question: was the production threshold (models/ensemble_config.json) chosen
on CALIBRATED probabilities? The app applies it to the calibrated value.

This script only READS your own saved files and prints them:
  1. the production threshold,
  2. the rows of models/*threshold*.csv nearest to it,
  3. every notebook line that mentions the threshold, so you can see
     which probabilities (raw ensemble or calibrated) it was tuned on.

    python scripts/check_threshold.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

cfg = json.loads((ROOT / "models" / "ensemble_config.json").read_text())
threshold = float(cfg["threshold"])
print(f"Production threshold in ensemble_config.json: {threshold}\n")

for name in ("threshold_results.csv", "calibrated_threshold_results.csv"):
    path = ROOT / "models" / name
    if not path.exists():
        print(f"[skip] {name} not found\n")
        continue

    df = pd.read_csv(path)
    print(f"=== {name}  (columns: {', '.join(df.columns)}) ===")

    col = next((c for c in df.columns if "threshold" in c.lower()), None)
    if col is None:
        print(df.head(10).to_string(index=False), "\n")
        continue

    nearest = df.iloc[(df[col] - threshold).abs().argsort()[:5]]
    print(nearest.sort_values(col).to_string(index=False), "\n")

print("=== Notebook lines that mention the threshold ===")
pattern = re.compile(r"threshold|0\.18", re.IGNORECASE)

for nb in sorted((ROOT / "notebooks").glob("*.ipynb")):
    cells = json.loads(nb.read_text(encoding="utf-8")).get("cells", [])
    hits = []
    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        for line in "".join(cell.get("source", [])).splitlines():
            if pattern.search(line) and ("0.18" in line or "best" in line.lower()
                                         or "optimal" in line.lower()
                                         or "calibrated" in line.lower()):
                hits.append(line.strip()[:140])
    if hits:
        print(f"\n--- {nb.name}")
        for h in hits[:12]:
            print("   ", h)

print("""
How to read this:
  - If the 0.18 row comes from calibrated_threshold_results.csv (and notebook
    17/18 selects it using calibrated probabilities), the app is correct.
  - If 0.18 was selected from threshold_results.csv using the RAW ensemble
    probabilities, then applying it to calibrated probabilities is a mismatch
    and the threshold must be re-selected on calibrated scores.
""")
