"""
CreditLens-AI
Plain-language names for model features

The model's columns are the Home Credit dataset's technical names
(``EXT_SOURCE_3``, ``BUREAU_RECENT_LOAN_DAYS``). People reading a
screenshot or a PDF need words.

Order of lookup:
1. exact, documented names (Home Credit column descriptions and the
   engineered features of the project's notebooks);
2. patterns (documents, missing-value flags, building statistics,
   history-table prefixes);
3. a word-by-word fallback that only translates words, it never invents
   a meaning.

Display only: nothing here changes a model input, a prediction or what
the API sends. ``friendly_name`` is pure and has no Streamlit imports.
"""

from __future__ import annotations

import re

# ---- 1. Exact names --------------------------------------------------

EXACT = {
    # Application (Home Credit column descriptions)
    "EXT_SOURCE_1": "External credit score 1",
    "EXT_SOURCE_2": "External credit score 2",
    "EXT_SOURCE_3": "External credit score 3",
    "EXT_SOURCE_MEAN": "External credit score (average of 3)",
    "EXT_SOURCE_MAX": "External credit score (highest of 3)",
    "EXT_SOURCE_MIN": "External credit score (lowest of 3)",
    "AMT_CREDIT": "Loan amount",
    "AMT_ANNUITY": "Regular loan payment",
    "AMT_GOODS_PRICE": "Price of goods financed",
    "AMT_INCOME_TOTAL": "Annual income",
    "CNT_CHILDREN": "Number of children",
    "CNT_FAM_MEMBERS": "Family members",
    "CODE_GENDER": "Gender",
    "NAME_CONTRACT_TYPE": "Type of loan",
    "NAME_EDUCATION_TYPE": "Education level",
    "NAME_FAMILY_STATUS": "Marital status",
    "NAME_HOUSING_TYPE": "Housing situation",
    "NAME_INCOME_TYPE": "Type of income",
    "NAME_TYPE_SUITE": "Who accompanied the applicant",
    "OCCUPATION_TYPE": "Occupation",
    "ORGANIZATION_TYPE": "Type of employer",
    "FLAG_OWN_CAR": "Owns a car",
    "FLAG_OWN_REALTY": "Owns a house or flat",
    "OWN_CAR_AGE": "Age of car (years)",
    "DAYS_BIRTH": "Age (days)",
    "DAYS_EMPLOYED": "Time in current job (days)",
    "DAYS_REGISTRATION": "Days since registration changed",
    "DAYS_ID_PUBLISH": "Days since ID document changed",
    "DAYS_LAST_PHONE_CHANGE": "Days since phone number changed",
    "REGION_POPULATION_RELATIVE": "Population density of home region",
    "REGION_RATING_CLIENT": "Home region rating",
    "REGION_RATING_CLIENT_W_CITY": "Home region and city rating",
    "HOUR_APPR_PROCESS_START": "Hour the application was made",
    "WEEKDAY_APPR_PROCESS_START": "Weekday of the application",
    "OBS_30_CNT_SOCIAL_CIRCLE": "Social circle: observations of 30-day overdue",
    "DEF_30_CNT_SOCIAL_CIRCLE": "Social circle: defaults on 30-day overdue",
    "OBS_60_CNT_SOCIAL_CIRCLE": "Social circle: observations of 60-day overdue",
    "DEF_60_CNT_SOCIAL_CIRCLE": "Social circle: defaults on 60-day overdue",
    "AMT_REQ_CREDIT_BUREAU_HOUR": "Credit enquiries in the last hour",
    "AMT_REQ_CREDIT_BUREAU_DAY": "Credit enquiries in the last day",
    "AMT_REQ_CREDIT_BUREAU_WEEK": "Credit enquiries in the last week",
    "AMT_REQ_CREDIT_BUREAU_MON": "Credit enquiries in the last month",
    "AMT_REQ_CREDIT_BUREAU_QRT": "Credit enquiries in the last quarter",
    "AMT_REQ_CREDIT_BUREAU_YEAR": "Credit enquiries in the last year",
    "FLAG_EMP_PHONE": "Work phone provided",
    "FLAG_WORK_PHONE": "Work landline provided",
    "FLAG_PHONE": "Home phone provided",
    "FLAG_MOBIL": "Mobile phone provided",
    "FLAG_EMAIL": "Email provided",
    "FLAG_CONT_MOBILE": "Mobile phone reachable",
    # Engineered (notebook 08)
    "AGE_YEARS": "Age (years)",
    "EMPLOYED_YEARS": "Years in current job",
    "EMPLOYMENT_TO_AGE_RATIO": "Share of life spent in current job",
    "CREDIT_TO_INCOME_RATIO": "Loan amount compared with income",
    "ANNUITY_TO_INCOME_RATIO": "Regular payment compared with income",
    "ANNUITY_TO_CREDIT_RATIO": "Regular payment compared with loan amount",
    "GOODS_TO_CREDIT_RATIO": "Goods price compared with loan amount",
    "INCOME_PER_FAMILY_MEMBER": "Income per family member",
    "CHILDREN_RATIO": "Children as a share of the family",
    "CREDIT_PER_FAMILY_MEMBER": "Loan amount per family member",
    "ANNUITY_PER_FAMILY_MEMBER": "Regular payment per family member",
    "TOTAL_DOCUMENTS": "Number of documents provided",
    "AGE_AT_EMPLOYMENT": "Age when the current job started",
    "CREDIT_GOODS_DIFF": "Loan amount minus goods price",
    "INCOME_MINUS_ANNUITY": "Income left after the regular payment",
    "LOAN_TERM": "Approximate loan term (payments)",
}

# ---- 3. Words --------------------------------------------------------

WORDS = {
    "TOTAL": "total", "MEAN": "average", "AVG": "average", "MAX": "highest",
    "MIN": "lowest", "STD": "variation in", "AMT": "amount",
    "CNT": "count", "NUM": "number", "RATIO": "ratio", "RATE": "rate",
    "DPD": "days past due", "DEF": "(with tolerance)",
    "PAYMENT": "payment", "PAYMENTS": "payments", "LOAN": "loan",
    "LOANS": "loans", "CREDIT": "credit", "DEBT": "debt", "LIMIT": "limit",
    "ANNUITY": "regular payment", "APPLICATION": "application",
    "APPLICATIONS": "applications", "DOWN": "down", "ACTIVE": "active",
    "CLOSED": "closed", "COMPLETED": "completed", "OVERDUE": "overdue",
    "DELAY": "delay", "RECENT": "recent", "OLD": "old", "OLDEST": "oldest",
    "FIRST": "first", "LAST": "last", "FUTURE": "future",
    "EXPIRED": "expired", "UPDATE": "update", "UPDATES": "updates",
    "HISTORY": "history", "APPROVED": "approved", "REFUSED": "refused",
    "CANCELED": "cancelled", "UNUSED": "unused", "APPROVAL": "approval",
    "REFUSAL": "refusal", "CANCEL": "cancellation", "MONTHS": "months",
    "BALANCE": "balance", "UTILIZATION": "utilisation",
    "RECEIVABLE": "receivable", "CARDS": "cards", "DAYS": "days",
    "DAY": "day", "SINCE": "since", "PER": "per", "TO": "to", "AGE": "age",
    "ENDDATE": "end date", "SEVERE": "severe", "BAD": "bad",
    "SOLD": "sold", "SIZE": "size", "EARLY": "early", "LATE": "late",
    "FULL": "full", "PARTIAL": "partial", "EXPECTED": "expected",
    "INSTALLMENTS": "instalments", "TERMINATION": "termination",
    "UNKNOWN": "unknown", "CURRENT": "current", "MONTH": "month",
    "APPLICATION_AMT": "application amount", "TERMINATION_": "termination",
    "365": "last 365 days", "MISSING": "not recorded",
    "LIVINGAREA": "living area", "LANDAREA": "land area",
    "BASEMENTAREA": "basement area", "COMMONAREA": "common area",
    "NONLIVINGAREA": "non-living area", "TOTALAREA": "total area",
    "APARTMENTS": "apartments", "LIVINGAPARTMENTS": "living apartments",
    "NONLIVINGAPARTMENTS": "non-living apartments",
    "ELEVATORS": "elevators", "ENTRANCES": "entrances",
    "FLOORSMAX": "maximum floors", "FLOORSMIN": "minimum floors",
    "YEARS": "years", "BEGINEXPLUATATION": "since building use began",
    "BUILD": "building age", "FONDKAPREMONT": "renovation fund",
    "HOUSETYPE": "house type", "WALLSMATERIAL": "wall material",
    "EMERGENCYSTATE": "emergency state",
}

PREFIXES = {
    "BUREAU_": "Credit bureau",
    "BB_": "Credit bureau monthly status",
    "PREV_": "Previous loan applications",
    "POS_": "Point-of-sale / cash loans",
    "INST_": "Instalment payments",
    "CC_": "Credit card",
}

# Building statistics suffixes (application table).
_BUILDING_SUFFIX = {"_AVG": "average", "_MODE": "most common", "_MEDI": "median"}

# ---- lookup ---------------------------------------------------------

_DOC = re.compile(r"^FLAG_DOCUMENT_(\d+)$")
_TOKEN = re.compile(r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b")


def _words(raw: str) -> str:
    out = []
    for token in raw.split("_"):
        out.append(WORDS.get(token, token.lower()))
    return " ".join(out)


def friendly_name(raw: object) -> str:
    """Plain-language label for a feature name; unknown names are tidied."""

    name = str(raw)

    if name in EXACT:
        return EXACT[name]

    match = _DOC.match(name)

    if match:
        return f"Document {match.group(1)} provided"

    if name.endswith("_MISSING"):
        return f"{friendly_name(name[:-8])} - not recorded"

    for prefix, label in PREFIXES.items():
        if name.startswith(prefix):
            text = _words(name[len(prefix):])
            return f"{label}: {text}"

    for suffix, label in _BUILDING_SUFFIX.items():
        if name.endswith(suffix):
            base = _words(name[: -len(suffix)])
            return f"Home building: {base} ({label})"

    if not name.replace("_", "").isupper():
        return name  # already readable, e.g. a category value

    text = _words(name)
    return text[:1].upper() + text[1:]


def replace_feature_names(text: str) -> str:
    """Swap technical feature names inside free text for plain labels."""

    if not text:
        return text

    return _TOKEN.sub(lambda m: friendly_name(m.group(0)), text)
