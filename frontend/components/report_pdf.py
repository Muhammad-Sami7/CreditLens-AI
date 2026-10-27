"""
CreditLens-AI
PDF Assessment Report

Builds a shareable PDF record of ONE assessment from the response the
API already returned (the same dictionary the page displays).

Nothing is calculated here: the decision, probability, threshold, SHAP
factors and Gemini text are copied exactly as received. The report only
lays them out.
"""

from __future__ import annotations

import io
from datetime import datetime
from typing import Any
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

INK = colors.HexColor("#0f172a")
MUTED = colors.HexColor("#64748b")
LINE = colors.HexColor("#e2e8f0")
RISK = colors.HexColor("#b91c1c")
SAFE = colors.HexColor("#15803d")

from components.feature_names import friendly_name, replace_feature_names  # noqa: E402
from components.formatting import display_value  # noqa: E402

TOP_SHAP_ROWS = 5

# The built-in PDF fonts only cover Latin-1. Replace the typographic
# characters AI text commonly contains so they never print as boxes.
_REPLACEMENTS = {
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "−": "-", "•": "-",
    "≥": ">=", "≤": "<=", "→": "->", "…": "...",
    " ": " ",
}


def _clean(value: Any) -> str:
    """Make text safe for the PDF: Latin-1 only, XML-escaped."""

    text = "" if value is None else replace_feature_names(str(value))

    for old, new in _REPLACEMENTS.items():
        text = text.replace(old, new)

    text = text.encode("latin-1", "replace").decode("latin-1")

    return escape(text)


def _pct(value: Any) -> str:
    return f"{float(value) * 100:.2f}%"


def _styles() -> dict[str, ParagraphStyle]:

    base = getSampleStyleSheet()["Normal"]

    def make(name: str, **kw: Any) -> ParagraphStyle:
        return ParagraphStyle(name, parent=base, **kw)

    return {
        "eyebrow": make("eyebrow", fontSize=8, textColor=MUTED,
                        fontName="Helvetica-Bold", spaceAfter=2),
        "title": make("title", fontSize=20, leading=24, textColor=INK,
                      fontName="Helvetica-Bold", spaceAfter=4),
        "h2": make("h2", fontSize=12, leading=15, textColor=INK,
                   fontName="Helvetica-Bold", spaceBefore=12, spaceAfter=5),
        "body": make("body", fontSize=9.5, leading=13.5, textColor=INK),
        "small": make("small", fontSize=8, leading=11, textColor=MUTED),
        "item": make("item", fontSize=9.5, leading=13, textColor=INK,
                     leftIndent=10, bulletIndent=0, spaceAfter=3),
        "label": make("label", fontSize=7.5, textColor=MUTED,
                      fontName="Helvetica-Bold"),
        "big": make("big", fontSize=15, leading=18, textColor=INK,
                    fontName="Helvetica-Bold"),
    }


def _bullets(items: list[str], style: ParagraphStyle) -> list[Paragraph]:
    return [Paragraph(_clean(i), style, bulletText="-") for i in items]


def _shap_table(
    factors: list[dict[str, Any]],
    colour: colors.Color,
    st: dict[str, ParagraphStyle],
) -> Table | None:

    factors = factors[:TOP_SHAP_ROWS]

    if not factors:
        return None

    rows = [["Field", "Value", "Models agreeing", "Contribution"]] + [
        [
            Paragraph(_clean(friendly_name(f["feature"])), st["body"]),
            Paragraph(
                _clean(display_value(
                    str(f["feature"]), f.get("feature_value")
                )),
                st["body"],
            ),
            (f"{int(f['models_agreeing'])} of 3"
             if f.get("models_agreeing") is not None else "-"),
            f"{float(f['shap_value']):+.4f}",
        ]
        for f in factors
    ]

    table = Table(rows, colWidths=[72 * mm, 40 * mm, 30 * mm, 28 * mm], repeatRows=1)
    table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 8),
        ("TEXTCOLOR", (0, 0), (-1, 0), MUTED),
        ("FONTSIZE", (2, 1), (3, -1), 9),
        ("TEXTCOLOR", (3, 1), (3, -1), colour),
        ("FONTNAME", (3, 1), (3, -1), "Helvetica-Bold"),
        ("ALIGN", (2, 0), (3, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, LINE),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))

    return table


def _footer(canvas: Any, doc: Any) -> None:

    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(
        18 * mm, 10 * mm,
        "CreditLens-AI - decision-support report. Not a credit decision.",
    )
    canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {doc.page}")
    canvas.restoreState()


def build_assessment_pdf(
    result: dict[str, Any],
    source: str = "Customer lookup",
) -> bytes:
    """Return the PDF (bytes) for one API assessment result."""

    st = _styles()
    explanation = result.get("llm_explanation") or {}
    shap = result.get("shap") or {}
    completeness = result.get("feature_completeness")
    is_high = str(result["risk_level"]).strip().lower() == "high risk"
    accent = RISK if is_high else SAFE

    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=16 * mm, bottomMargin=18 * mm,
        title=f"CreditLens-AI assessment - {result['customer_id']}",
        author="CreditLens-AI",
    )

    story: list[Any] = [
        Paragraph("CREDITLENS-AI  |  CREDIT RISK ASSESSMENT REPORT", st["eyebrow"]),
        Paragraph(f"Customer {_clean(result['customer_id'])}", st["title"]),
        Paragraph(
            f"Generated {datetime.now().strftime('%d %b %Y, %H:%M')}"
            + (f"  |  Assessment #{result['prediction_id']}"
               if result.get("prediction_id") is not None else "")
            + f"  |  Input: {_clean(source)}",
            st["small"],
        ),
        Spacer(1, 6),
        HRFlowable(width="100%", color=LINE, thickness=0.8),
        Spacer(1, 8),
    ]

    # ---- Decision summary --------------------------------------

    decision_rows = [[
        [Paragraph("RISK LEVEL", st["label"]),
         Paragraph(f'<font color="{accent.hexval().replace("0x", "#")}">'
                   f'{_clean(result["risk_level"]).upper()}</font>', st["big"])],
        [Paragraph("ESTIMATED DEFAULT RISK", st["label"]),
         Paragraph(_pct(result["probability"]), st["big"])],
        [Paragraph("REVIEW THRESHOLD", st["label"]),
         Paragraph(_pct(result["threshold"]), st["big"])],
    ]]

    box = Table(decision_rows, colWidths=[58 * mm, 58 * mm, 58 * mm])
    box.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.8, LINE),
        ("LINEBEFORE", (0, 0), (0, -1), 3, accent),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))

    story += [
        box,
        Spacer(1, 6),
        Paragraph(
            f"<b>Decision:</b> {_clean(result['decision'])}. The estimated "
            f"default risk of {_pct(result['probability'])} is "
            f"{'at or above' if is_high else 'below'} the review threshold "
            f"of {_pct(result['threshold'])}.",
            st["body"],
        ),
    ]

    if completeness and completeness.get("missing_count", 0) > 0:

        note = (
            f"<b>Data note:</b> {completeness['missing_count']:,} of "
            f"{completeness['expected_count']:,} data fields were not "
            "provided. Missing fields are treated as 'no data', which "
            "can make this result less reliable."
        )

        story += [Spacer(1, 6), Paragraph(note, st["body"])]

    # ---- Explanation -------------------------------------------

    if explanation.get("summary"):
        story += [Paragraph("AI-assisted summary", st["h2"]),
                  Paragraph(_clean(explanation["summary"]), st["body"])]

    if explanation.get("risk_explanation"):
        story += [Paragraph("Risk explanation", st["h2"]),
                  Paragraph(_clean(explanation["risk_explanation"]), st["body"])]

    # ---- SHAP evidence -----------------------------------------

    positive = shap.get("positive_factors") or []
    negative = shap.get("negative_factors") or []

    if positive or negative:
        story.append(Paragraph("What drove the model's score", st["h2"]))

        pos_table = _shap_table(positive, RISK, st)
        if pos_table:
            story += [Paragraph("Factors increasing modeled risk", st["label"]),
                      Spacer(1, 3), pos_table, Spacer(1, 8)]

        neg_table = _shap_table(negative, SAFE, st)
        if neg_table:
            story += [Paragraph("Factors supporting lower modeled risk", st["label"]),
                      Spacer(1, 3), neg_table, Spacer(1, 4)]

        story.append(Paragraph(
            "Contributions combine the LightGBM, XGBoost and CatBoost models "
            "using the production ensemble weights. They show how the models "
            "scored this customer; they do not prove that a field caused the "
            "outcome.", st["small"]))

    for title, items in (
        ("Key risk factors (AI-written)", explanation.get("key_risk_factors")),
        ("Protective factors (AI-written)", explanation.get("protective_factors")),
        ("Recommended analyst actions", explanation.get("analyst_actions")),
    ):
        if items:
            story += [Paragraph(title, st["h2"]), *_bullets(items, st["item"])]

    # ---- Model detail ------------------------------------------

    probs = result.get("model_probabilities") or {}
    weights = result.get("model_weights") or {}

    def _w(model: str) -> str:
        """' (weight 35%)' when the API supplied the ensemble weights."""
        value = weights.get(model)
        return f" (weight {float(value) * 100:.0f}%)" if value is not None else ""

    detail = [["Stage", "Probability"],
              [f"LightGBM{_w('lightgbm')}", _pct(probs.get("lightgbm", 0))],
              [f"XGBoost{_w('xgboost')}", _pct(probs.get("xgboost", 0))],
              [f"CatBoost{_w('catboost')}", _pct(probs.get("catboost", 0))],
              ["Weighted ensemble (uncalibrated)", _pct(result["ensemble_probability"])],
              ["After isotonic calibration (used for the decision)",
               _pct(result["probability"])]]

    dt = Table(detail, colWidths=[110 * mm, 40 * mm])
    dt.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TEXTCOLOR", (0, 0), (-1, 0), MUTED),
        ("ALIGN", (1, 0), (1, -1), "RIGHT"),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, LINE),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))

    story += [KeepTogether([
                  Paragraph("Technical detail: how each model scored", st["h2"]),
                  dt]),
              Spacer(1, 14),
              HRFlowable(width="100%", color=LINE, thickness=0.8),
              Spacer(1, 4),
              Paragraph(
                  "This report supports, and does not replace, the loan "
                  "officer's judgement. The AI-written text is generated from "
                  "the model's evidence and is not an independent credit "
                  "decision.", st["small"])]

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)

    return buffer.getvalue()
