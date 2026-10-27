"""
CreditLens-AI
Production Frontend
"""

from __future__ import annotations

import streamlit as st
from components.markup import render_html

from config import (
    APP_DESCRIPTION,
    APP_NAME,
    APP_VERSION,
    NAVIGATION_ITEMS,
)
from pages.about import render_about
from pages.customer_history import (
    render_customer_history,
)
from pages.dashboard import (
    render_dashboard,
)
from pages.risk_assessment import (
    render_risk_assessment,
)


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title=APP_NAME,
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Global CSS
# ============================================================

render_html(
    """
    <style>

    @import url("https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@600;700;800&display=swap");

    /* ========================================================
       Base
       ======================================================== */

    html,
    body,
    [class*="css"] {

        font-family:
            Inter,
            -apple-system,
            BlinkMacSystemFont,
            "Segoe UI",
            sans-serif;

    }

    .stApp {

        background:
            #F5F7FA;

        color:
            #172033;

    }

    /* Remove Streamlit header decoration */

    header {

        background:
            transparent !important;

    }

    [data-testid="stHeader"] {

        background:
            transparent;

    }

    /* ========================================================
       Sidebar
       ======================================================== */

    [data-testid="stSidebar"] {

        background:
            #0F172A;

        border-right:
            1px solid #1E293B;

    }

    [data-testid="stSidebar"] > div:first-child {

        padding:
            28px 18px;

    }

    .sidebar-brand {

        padding:
            10px 8px 28px 8px;

    }

    .sidebar-brand-name {

        color:
            #FFFFFF;

        font-family:
            "Plus Jakarta Sans", Inter, "Segoe UI", sans-serif;

        font-size:
            30px;

        font-weight:
            800;

        letter-spacing:
            -0.8px;

        line-height:
            1.1;

    }

    .sidebar-brand-name .brand-ai {

        color:
            #60A5FA;

        font-weight:
            700;

    }

    .sidebar-brand-subtitle {

        color:
            #94A3B8;

        font-size:
            12px;

        margin-top:
            5px;

        line-height:
            1.5;

    }

    .sidebar-divider {

        height:
            1px;

        background:
            #1E293B;

        margin:
            4px 0 24px 0;

    }

    .sidebar-section {

        color:
            #64748B;

        font-size:
            10px;

        font-weight:
            700;

        letter-spacing:
            1.2px;

        text-transform:
            uppercase;

        margin:
            0 8px 10px 8px;

    }

    /* ========================================================
       Custom Navigation
       ======================================================== */

    div[data-testid="stRadio"] label {

        color:
            #CBD5E1 !important;

        font-size:
            14px !important;

        font-weight:
            500 !important;

        padding:
            9px 10px !important;

        border-radius:
            7px;

    }

    div[data-testid="stRadio"] label:hover {

        background:
            #1E293B;

        color:
            #FFFFFF !important;

    }

    div[data-testid="stRadio"] > div {

        gap:
            3px;

    }

    /* Hide radio circles */

    div[data-testid="stRadio"] label > div:first-child {

        display:
            none;

    }

    /* ========================================================
       Main Content
       ======================================================== */

    .main .block-container {

        max-width:
            1420px;

        padding:
            42px 56px 70px 56px;

    }

    /* ========================================================
       Page Header
       ======================================================== */

    .page-eyebrow {

        color:
            #2563EB;

        font-size:
            11px;

        font-weight:
            700;

        letter-spacing:
            1.6px;

        margin-bottom:
            9px;

    }

    .page-title {

        color:
            #0F172A;

        font-size:
            34px;

        line-height:
            1.15;

        font-weight:
            750;

        letter-spacing:
            -1.2px;

        margin-bottom:
            9px;

    }

    .page-description {

        color:
            #64748B;

        font-size:
            14px;

        line-height:
            1.6;

        max-width:
            720px;

        margin-bottom:
            28px;

    }

    /* ========================================================
       Panels
       ======================================================== */

    .panel {

        background:
            #FFFFFF;

        border:
            1px solid #E2E8F0;

        border-radius:
            14px;

        padding:
            24px;

        box-shadow:
            0 1px 2px rgba(15, 23, 42, 0.03);

    }

    .section-header {

        margin-bottom:
            15px;

    }

    .section-label {

        color:
            #64748B;

        font-size:
            10px;

        font-weight:
            700;

        letter-spacing:
            1.2px;

        margin-bottom:
            5px;

    }

    .section-title {

        color:
            #0F172A;

        font-size:
            20px;

        font-weight:
            700;

        letter-spacing:
            -0.4px;

    }

    .section-gap {

        height:
            22px;

    }

    /* ========================================================
       Status Strip
       ======================================================== */

    .status-strip {

        display:
            flex;

        justify-content:
            space-between;

        align-items:
            center;

        background:
            #FFFFFF;

        border:
            1px solid #E2E8F0;

        border-radius:
            10px;

        padding:
            12px 16px;

        margin-bottom:
            22px;

        color:
            #334155;

        font-size:
            13px;

    }

    .status-dot {

        display:
            inline-block;

        width:
            7px;

        height:
            7px;

        border-radius:
            50%;

        background:
            #16A34A;

        margin-right:
            8px;

    }

    .status-meta {

        color:
            #64748B;

        font-size:
            12px;

    }

    /* ========================================================
       KPI Cards
       ======================================================== */

    .kpi-card {

        background:
            #FFFFFF;

        border:
            1px solid #E2E8F0;

        border-radius:
            12px;

        padding:
            20px;

        min-height:
            130px;

        box-shadow:
            0 1px 2px rgba(15, 23, 42, 0.03);

    }

    .kpi-value {

        color:
            #0F172A;

        font-size:
            28px;

        font-weight:
            750;

        letter-spacing:
            -0.8px;

        margin-top:
            8px;

    }

    .kpi-caption {

        color:
            #64748B;

        font-size:
            12px;

        margin-top:
            5px;

        line-height:
            1.4;

    }

    .metric-label {

        color:
            #64748B;

        font-size:
            10px;

        font-weight:
            700;

        letter-spacing:
            0.8px;

        text-transform:
            uppercase;

    }

    .metric-value {

        color:
            #0F172A;

        font-size:
            18px;

        font-weight:
            700;

        margin-top:
            6px;

    }

    /* ========================================================
       Pipeline
       ======================================================== */

    .pipeline {

        display:
            flex;

        align-items:
            center;

        gap:
            10px;

        margin-top:
            25px;

        flex-wrap:
            wrap;

    }

    .pipeline-step {

        background:
            #F8FAFC;

        border:
            1px solid #E2E8F0;

        border-radius:
            8px;

        padding:
            13px 15px;

        color:
            #334155;

        font-size:
            12px;

        font-weight:
            600;

    }

    .pipeline-step span {

        color:
            #2563EB;

        margin-right:
            7px;

        font-size:
            10px;

    }

    .pipeline-arrow {

        color:
            #94A3B8;

    }

    /* ========================================================
       Model Rows
       ======================================================== */

    .model-row {

        display:
            grid;

        grid-template-columns:
            1fr 70px 100px;

        align-items:
            center;

        border-bottom:
            1px solid #E2E8F0;

        padding:
            14px 0;

    }

    .model-row:last-child {

        border-bottom:
            none;

    }

    .model-name {

        color:
            #334155;

        font-size:
            13px;

        font-weight:
            600;

    }

    .model-weight {

        color:
            #94A3B8;

        font-size:
            12px;

    }

    .model-probability {

        color:
            #0F172A;

        font-size:
            13px;

        font-weight:
            700;

        text-align:
            right;

    }

    /* ========================================================
       Weight Row (Ensemble Configuration)
       ======================================================== */

    .weight-row {

        display:
            flex;

        justify-content:
            space-between;

        align-items:
            center;

        border-bottom:
            1px solid #E2E8F0;

        padding:
            13px 0;

        color:
            #334155;

        font-size:
            13px;

    }

    .weight-row:last-child {

        border-bottom:
            none;

    }

    .weight-row strong {

        color:
            #0F172A;

    }

    /* ========================================================
       Prediction Card
       ======================================================== */

    .prediction-card {

        background:
            #FFFFFF;

        border:
            1px solid #E2E8F0;

        border-radius:
            16px;

        padding:
            28px;

        box-shadow:
            0 4px 12px rgba(15, 23, 42, 0.04);

    }

    .risk-low {

        border-left:
            5px solid #15803D;

    }

    .risk-high {

        border-left:
            5px solid #B91C1C;

    }

    .prediction-card-header {

        display:
            flex;

        justify-content:
            space-between;

        align-items:
            flex-start;

    }

    .prediction-status {

        color:
            #0F172A;

        font-size:
            30px;

        font-weight:
            750;

        margin-top:
            5px;

    }

    .prediction-score {

        color:
            #0F172A;

        font-size:
            38px;

        font-weight:
            750;

        letter-spacing:
            -1px;

    }

    .prediction-divider {

        height:
            1px;

        background:
            #E2E8F0;

        margin:
            22px 0;

    }

    .prediction-grid {

        display:
            grid;

        grid-template-columns:
            repeat(3, 1fr);

        gap:
            25px;

    }

    .prediction-description {

        color:
            #64748B;

        font-size:
            13px;

        line-height:
            1.6;

        margin-top:
            20px;

    }

    /* ========================================================
       Profile
       ======================================================== */

    .profile-card {

        display:
            grid;

        grid-template-columns:
            repeat(3, 1fr);

        gap:
            14px;

    }

    .profile-item {

        background:
            #FFFFFF;

        border:
            1px solid #E2E8F0;

        border-radius:
            10px;

        padding:
            17px;

    }

    .profile-value {

        color:
            #0F172A;

        font-size:
            15px;

        font-weight:
            650;

        margin-top:
            7px;

    }

    /* ========================================================
       Recommendation
       ======================================================== */

    .recommendation-panel {

        border-radius:
            12px;

        padding:
            22px;

        border:
            1px solid #E2E8F0;

    }

    .recommendation-low {

        background:
            #F0FDF4;

        border-color:
            #BBF7D0;

    }

    .recommendation-high {

        background:
            #FEF2F2;

        border-color:
            #FECACA;

    }

    .recommendation-title {

        color:
            #0F172A;

        font-size:
            19px;

        font-weight:
            700;

        margin-top:
            8px;

    }

    .recommendation-text {

        color:
            #475569;

        font-size:
            13px;

        line-height:
            1.6;

        margin-top:
            8px;

    }

    /* ========================================================
       Explainability / Empty States
       ======================================================== */

    .empty-panel {

        background:
            #F8FAFC;

        border:
            1px dashed #CBD5E1;

        border-radius:
            10px;

        padding:
            24px;

    }

    .empty-panel-title {

        color:
            #334155;

        font-size:
            15px;

        font-weight:
            650;

    }

    .empty-panel-text {

        color:
            #64748B;

        font-size:
            13px;

        line-height:
            1.6;

        margin-top:
            7px;

    }

    .history-empty {

        background:
            #FFFFFF;

        border:
            1px solid #E2E8F0;

        border-radius:
            16px;

        padding:
            80px 40px;

        text-align:
            center;

        box-shadow:
            0 1px 2px rgba(15, 23, 42, 0.03);

    }

    .empty-icon {

        display:
            inline-flex;

        align-items:
            center;

        justify-content:
            center;

        width:
            48px;

        height:
            48px;

        border-radius:
            10px;

        background:
            #EFF6FF;

        color:
            #2563EB;

        font-size:
            13px;

        font-weight:
            750;

        margin-bottom:
            18px;

    }

    .empty-title {

        color:
            #0F172A;

        font-size:
            20px;

        font-weight:
            700;

    }

    .empty-text {

        color:
            #64748B;

        font-size:
            13px;

        line-height:
            1.6;

        max-width:
            550px;

        margin:
            10px auto 0 auto;

    }

    .empty-status {

        color:
            #2563EB;

        font-size:
            10px;

        font-weight:
            700;

        letter-spacing:
            1px;

        margin-top:
            22px;

    }

    /* ========================================================
       Inputs
       ======================================================== */

    .stTextInput input,
    .stTextArea textarea {

        border:
            1px solid #CBD5E1 !important;

        border-radius:
            8px !important;

        background:
            #FFFFFF !important;

        color:
            #172033 !important;

    }

    .stTextInput input:focus,
    .stTextArea textarea:focus {

        border-color:
            #2563EB !important;

        box-shadow:
            0 0 0 1px #2563EB !important;

    }

    /* ========================================================
       Buttons
       ======================================================== */

    .stButton > button {

        background:
            #0F172A !important;

        color:
            #FFFFFF !important;

        border:
            1px solid #0F172A !important;

        border-radius:
            8px !important;

        font-weight:
            650 !important;

        padding:
            9px 18px !important;

    }

    .stButton > button:hover {

        background:
            #1E293B !important;

        border-color:
            #1E293B !important;

    }

    [data-testid="stAppDeployButton"],
    .stAppDeployButton,
    .stDeployButton {

        display:
            none !important;

    }

    .stDownloadButton > button {

        background:
            linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;

        color:
            #FFFFFF !important;

        border:
            1px solid #1D4ED8 !important;

        border-radius:
            8px !important;

        font-weight:
            650 !important;

        padding:
            9px 18px !important;

        box-shadow:
            0 2px 6px rgba(37, 99, 235, 0.28) !important;

    }

    .stDownloadButton > button:hover {

        background:
            linear-gradient(135deg, #1D4ED8 0%, #1E40AF 100%) !important;

        border-color:
            #1E40AF !important;

        color:
            #FFFFFF !important;

    }

    .stDownloadButton > button p {

        color:
            #FFFFFF !important;

    }

    .page-title {

        font-family:
            "Plus Jakarta Sans", Inter, "Segoe UI", sans-serif !important;

        letter-spacing:
            -0.6px;

    }

    /* ========================================================
       Expander
       ======================================================== */

    [data-testid="stExpander"] {

        border:
            1px solid #E2E8F0 !important;

        border-radius:
            10px !important;

        background:
            #FFFFFF !important;

    }

    /* ========================================================
       Probability / Threshold Gauge
       ======================================================== */

    .gauge-wrap {

        position:
            relative;

        height:
            104px;

        margin:
            4px 0 4px 0;

    }

    .gauge-track {

        position:
            absolute;

        top:
            44px;

        left:
            0;

        right:
            0;

        height:
            10px;

        background:
            #E2E8F0;

        border-radius:
            6px;

    }

    .gauge-fill {

        position:
            absolute;

        top:
            0;

        left:
            0;

        height:
            100%;

        border-radius:
            6px;

    }

    .gauge-fill-low {

        background:
            #15803D;

    }

    .gauge-fill-high {

        background:
            #B91C1C;

    }

    .gauge-scale {

        display:
            flex;

        justify-content:
            space-between;

        margin-top:
            6px;

        color:
            #94A3B8;

        font-size:
            11px;

    }

    .gauge-marker {

        position:
            absolute;

        transform:
            translateX(-50%);

        text-align:
            center;

    }

    .gauge-marker-label {

        display:
            inline-block;

        font-size:
            11px;

        font-weight:
            700;

        padding:
            3px 8px;

        border-radius:
            6px;

        white-space:
            nowrap;

    }

    .gauge-marker-probability {

        top:
            0;

    }

    .gauge-marker-probability .gauge-marker-label {

        background:
            #0F172A;

        color:
            #FFFFFF;

    }

    .gauge-marker-probability .gauge-marker-line {

        width:
            2px;

        height:
            20px;

        margin:
            4px auto 0 auto;

        background:
            #0F172A;

    }

    .gauge-edge-start .gauge-marker-label {

        transform: translateX(calc(50% - 1px));

    }

    .gauge-edge-end .gauge-marker-label {

        transform: translateX(calc(-50% + 1px));

    }

    .gauge-marker-threshold {

        top:
            54px;

    }

    .gauge-marker-threshold .gauge-marker-line {

        width:
            2px;

        height:
            16px;

        margin:
            0 auto 4px auto;

        background:
            #94A3B8;

    }

    .gauge-marker-threshold .gauge-marker-label {

        background:
            #FFFFFF;

        color:
            #334155;

        border:
            1px solid #94A3B8;

    }

    .gauge-interpretation {

        margin-top:
            24px;

        padding:
            14px 16px;

        border-radius:
            10px;

        border:
            1px solid;

        font-size:
            13px;

        line-height:
            1.65;

    }

    .gauge-interpretation-low {

        background:
            #F0FDF4;

        border-color:
            #BBF7D0;

        color:
            #14532D;

    }

    .gauge-interpretation-high {

        background:
            #FEF2F2;

        border-color:
            #FECACA;

        color:
            #7F1D1D;

    }

    /* ========================================================
       Risk / Protective Factor List
       ======================================================== */

    .factor-section-title {

        color:
            #64748B;

        font-size:
            10px;

        font-weight:
            700;

        letter-spacing:
            1px;

        text-transform:
            uppercase;

        margin:
            18px 0 10px 0;

    }

    .factor-list {

        display:
            flex;

        flex-direction:
            column;

        gap:
            8px;

    }

    .factor-item {

        display:
            flex;

        align-items:
            flex-start;

        gap:
            10px;

        padding:
            12px 14px;

        border-radius:
            10px;

        border:
            1px solid;

        font-size:
            13px;

        line-height:
            1.55;

    }

    .factor-item-risk {

        background:
            #FEF2F2;

        border-color:
            #FECACA;

        color:
            #7F1D1D;

    }

    .factor-item-protective {

        background:
            #F0FDF4;

        border-color:
            #BBF7D0;

        color:
            #14532D;

    }

    .factor-marker {

        flex-shrink:
            0;

        margin-top:
            1px;

        font-size:
            10px;

        font-weight:
            700;

        padding:
            2px 7px;

        border-radius:
            5px;

    }

    .factor-marker-risk {

        background:
            #B91C1C;

        color:
            #FFFFFF;

    }

    .factor-marker-protective {

        background:
            #15803D;

        color:
            #FFFFFF;

    }

    /* ========================================================
       AI Explanation Panel
       ======================================================== */

    .explanation-summary {

        color:
            #172033;

        font-size:
            15px;

        font-weight:
            500;

        line-height:
            1.7;

    }

    .explanation-block-title {

        color:
            #64748B;

        font-size:
            10px;

        font-weight:
            700;

        letter-spacing:
            1px;

        text-transform:
            uppercase;

        margin:
            20px 0 8px 0;

    }

    .explanation-text {

        color:
            #334155;

        font-size:
            13px;

        line-height:
            1.7;

    }

    .analyst-actions-list {

        margin:
            0;

        padding-left:
            0;

        list-style:
            none;

        display:
            flex;

        flex-direction:
            column;

        gap:
            9px;

    }

    .analyst-action-item {

        display:
            flex;

        gap:
            10px;

        align-items:
            flex-start;

        font-size:
            13px;

        line-height:
            1.6;

        color:
            #172033;

    }

    .analyst-action-index {

        background:
            #0F172A;

        color:
            #FFFFFF;

        font-size:
            11px;

        font-weight:
            700;

        width:
            20px;

        height:
            20px;

        min-width:
            20px;

        border-radius:
            50%;

        display:
            flex;

        align-items:
            center;

        justify-content:
            center;

        margin-top:
            1px;

    }

    .explanation-disclaimer {

        margin-top:
            18px;

        padding-top:
            14px;

        border-top:
            1px solid #E2E8F0;

        color:
            #94A3B8;

        font-size:
            11px;

        line-height:
            1.6;

    }

    /* ========================================================
       Footer / Misc
       ======================================================== */

    /* ========================================================
       SHAP Contribution Rows
       ======================================================== */

    .shap-row {

        display: grid;

        grid-template-columns: 1.3fr 0.7fr 1.2fr;

        gap: 12px;

        align-items: center;

        padding: 9px 0;

        border-bottom: 1px solid #E2E8F0;

        font-size: 13px;

    }

    .shap-row:last-child {

        border-bottom: none;

    }

    .shap-name {

        color: #0F172A;

        font-weight: 600;

        word-break: break-word;

    }

    .shap-value {

        color: #64748B;

        font-size: 12px;

    }

    .shap-bar-track {

        background: #F1F5F9;

        border-radius: 4px;

        height: 8px;

        position: relative;

    }

    .shap-bar {

        height: 100%;

        border-radius: 4px;

    }

    .shap-bar-risk {

        background: #B91C1C;

    }

    .shap-bar-protective {

        background: #15803D;

    }

    .shap-contribution {

        color: #475569;

        font-size: 11px;

        margin-top: 3px;

    }

    /* ========================================================
       Customer History Timeline
       ======================================================== */

    .history-item {

        display: grid;

        grid-template-columns: 150px 1fr auto;

        gap: 18px;

        align-items: center;

        background: #FFFFFF;

        border: 1px solid #E2E8F0;

        border-left-width: 5px;

        border-radius: 12px;

        padding: 16px 20px;

        margin-bottom: 10px;

    }

    .history-item-low {

        border-left-color: #15803D;

    }

    .history-item-high {

        border-left-color: #B91C1C;

    }

    .history-level {

        color: #0F172A;

        font-size: 15px;

        font-weight: 700;

    }

    .history-meta {

        color: #64748B;

        font-size: 12px;

        line-height: 1.6;

    }

    .history-probability {

        color: #0F172A;

        font-size: 22px;

        font-weight: 750;

        text-align: right;

    }

    /* ========================================================
       Risk Distribution Bar
       ======================================================== */

    .dist-bar {

        display: flex;

        height: 14px;

        border-radius: 7px;

        overflow: hidden;

        background: #E2E8F0;

        margin-top: 14px;

    }

    .dist-high {

        background: #B91C1C;

    }

    .dist-low {

        background: #15803D;

    }

    .dist-legend {

        display: flex;

        justify-content: space-between;

        color: #475569;

        font-size: 12px;

        margin-top: 10px;

    }

    /* Sidebar nav labels: newer Streamlit versions render the text
       in nested elements that ignore the label colour. */

    [data-testid="stSidebar"] div[data-testid="stRadio"] label p,
    [data-testid="stSidebar"] div[data-testid="stRadio"] label span,
    [data-testid="stSidebar"] div[data-testid="stRadio"] label div {

        color:
            #CBD5E1 !important;

    }

    [data-testid="stSidebar"] div[data-testid="stRadio"] label:hover p,
    [data-testid="stSidebar"] div[data-testid="stRadio"] label:hover span,
    [data-testid="stSidebar"] div[data-testid="stRadio"] label:hover div,
    [data-testid="stSidebar"] div[data-testid="stRadio"] label:has(input:checked) p {

        color:
            #FFFFFF !important;

    }

    /* Hide Streamlit's automatic multipage nav: navigation is
       handled by the custom sidebar; files in pages/ only
       define render functions and are not standalone pages. */

    [data-testid="stSidebarNav"] {

        display:
            none;

    }

    #MainMenu {

        visibility:
            hidden;

    }

    footer {

        visibility:
            hidden;

    }

    </style>
    """)


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:

    render_html(
        """
        <div class="sidebar-brand">

            <div class="sidebar-brand-name">
                CreditLens<span class="brand-ai">-AI</span>
            </div>

            <div class="sidebar-brand-subtitle">
                Financial Risk Intelligence
            </div>

        </div>

        <div class="sidebar-divider"></div>

        <div class="sidebar-section">
            Platform
        </div>
        """)

    selected_page = st.radio(
        "Navigation",
        NAVIGATION_ITEMS,
        label_visibility="collapsed",
    )

    render_html(
        """
        <div style="height: 35vh;"></div>

        <div class="sidebar-divider"></div>

        <div style="
            color:#64748B;
            font-size:11px;
            line-height:1.6;
            padding:0 8px;
        ">
            Credit decision-support prototype
            <br>
            Built on the public Home Credit dataset
        </div>
        """)


# ============================================================
# Page Routing
# ============================================================

if selected_page == "Dashboard":

    render_dashboard()

elif selected_page == "Risk Assessment":

    render_risk_assessment()

elif selected_page == "About":

    render_about()

elif selected_page == "Customer History":

    render_customer_history()