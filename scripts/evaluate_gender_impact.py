"""
Does the model treat men and women differently? (read-only, YOUR data)

CODE_GENDER is a column of the public Home Credit dataset that the
trained model uses as an input. This script measures what that means in
practice, so the project can state it with numbers instead of a worry.

For a sample of customers (data/processed/train_processed.parquet, which
has TARGET) it reports, per gender group:
  customers, ACTUAL default rate, mean predicted probability,
  share flagged at the production threshold, recall, AUC

and then re-scores everyone with gender set to the same value for all
(neutralised) to show how much the model's quality and the flag rates
change when gender cannot influence the score.

Reading the result:
  * Different default rates between groups are a property of the data.
  * If neutralising gender barely changes AUC and the flag rates, the
    feature adds almost nothing and can be dropped at the next retrain.
  * This is an audit, not a fix: nothing in the production model changes.

    python scripts\\evaluate_gender_impact.py
    python scripts\\evaluate_gender_impact.py --n 60000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.predictor import CreditLensPredictor  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=40000)
parser.add_argument(
    "--data", default=str(ROOT / "data" / "processed" / "train_processed.parquet")
)
args = parser.parse_args()

data_path = Path(args.data)

if not data_path.exists():
    sys.exit(f"Missing {data_path}")

frame = pd.read_parquet(data_path)

if "TARGET" not in frame.columns:
    sys.exit("The data file needs a TARGET column.")

if "CODE_GENDER" not in frame.columns:
    sys.exit("CODE_GENDER is not a column of this data file.")

if len(frame) > args.n:
    frame = frame.sample(args.n, random_state=42)

# The training encoder numbered the categories alphabetically
# (F, M, XNA). Used only to print readable group names.
labels = ["F", "M", "XNA"]

predictor = CreditLensPredictor()
columns = list(predictor.feature_columns)
threshold = float(predictor.ensemble.threshold)

y = frame["TARGET"].to_numpy()
codes = frame["CODE_GENDER"].to_numpy()
X = frame[columns].astype(float)


def score(matrix: pd.DataFrame) -> np.ndarray:
    return np.asarray(predictor.predict(matrix)["probability"], dtype=float)


def group_name(code: float) -> str:
    index = int(code)
    return labels[index] if 0 <= index < len(labels) else f"code {index}"


def report(title: str, probability: np.ndarray) -> None:
    print(f"\n{title}")
    header = (f"{'group':<8}{'n':>8}{'actual':>9}{'mean pred':>11}"
              f"{'flagged':>9}{'recall':>8}{'AUC':>7}")
    print(header)
    print("-" * len(header))

    for code in sorted(np.unique(codes)):
        mask = codes == code
        if mask.sum() < 200 or len(np.unique(y[mask])) < 2:
            continue
        flagged = probability[mask] >= threshold
        recall = flagged[y[mask] == 1].mean()
        print(f"{group_name(code):<8}{mask.sum():>8,}{y[mask].mean():>9.2%}"
              f"{probability[mask].mean():>11.2%}{flagged.mean():>9.1%}"
              f"{recall:>8.1%}{roc_auc_score(y[mask], probability[mask]):>7.3f}")

    print(f"{'ALL':<8}{len(y):>8,}{y.mean():>9.2%}{probability.mean():>11.2%}"
          f"{(probability >= threshold).mean():>9.1%}"
          f"{(probability >= threshold)[y == 1].mean():>8.1%}"
          f"{roc_auc_score(y, probability):>7.3f}")


print(f"Sample: {len(frame):,} customers, threshold {threshold}")

report("A. Production model as it is", score(X))

# Neutralised: everyone gets the most common gender value.
common = pd.Series(codes).mode().iloc[0]
neutral = X.copy()
neutral["CODE_GENDER"] = float(common)

report(f"B. Gender neutralised (everyone = '{group_name(common)}')",
       score(neutral))

print("""
How to read it: compare the AUC in 'ALL' for A and B. A difference near
zero means gender adds almost nothing and can be removed when the model
is next retrained. Compare 'flagged' between groups in A and B to see
whether gender changes who is flagged. Quote these numbers; do not guess.
""")
