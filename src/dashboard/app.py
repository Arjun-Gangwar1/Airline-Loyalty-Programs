"""
Dashboard data loader and shared constants for the Streamlit app.

Import these helpers into dashboard.py to avoid raw file I/O in the UI layer.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import streamlit as st

SEGMENT_COLORS = {
    'Premium Loyalists':    '#2ecc71',
    'Silent Drifters':      '#e74c3c',
    'Miles Hoarders':       '#f39c12',
    'Seasonal Travelers':   '#3498db',
    'Rising Stars':         '#9b59b6',
    'Budget Frequent Flyers': '#1abc9c',
    'At-Risk VIPs':         '#c0392b',
}

RISK_COLORS = {
    'low':    '#2ecc71',
    'medium': '#f39c12',
    'high':   '#e74c3c',
}


@st.cache_data
def load_dashboard_data() -> dict:
    """Load all CSVs and JSON files needed by the dashboard. Returns a dict."""
    base = Path("data/final")
    reports = Path("outputs/reports")

    data: dict = {}

    # Customer segments + predictions
    for key, path in [
        ("segments",    base    / "customer_features_segmented.csv"),
        ("predictions", reports / "retention_actions.csv"),
        ("model_comparison", reports / "model_comparison.csv"),
    ]:
        try:
            data[key] = pd.read_csv(path)
        except FileNotFoundError:
            data[key] = pd.DataFrame()

    # Feature importance
    try:
        data["feature_importance"] = pd.read_csv(reports / "feature_importance.csv")
    except FileNotFoundError:
        data["feature_importance"] = pd.DataFrame()

    # Pipeline summary
    for json_path in [
        reports / "pipeline_summary.json",
        Path("checkpoints") / "progress.json",
    ]:
        try:
            with open(json_path) as f:
                data["pipeline_summary"] = json.load(f)
            break
        except FileNotFoundError:
            data["pipeline_summary"] = {}

    # Retention playbook
    playbook_path = reports / "retention_playbook.json"
    try:
        with open(playbook_path) as f:
            data["playbook"] = json.load(f)
    except FileNotFoundError:
        data["playbook"] = {}

    return data


def kpi_card(label: str, value: str, delta: str = "", color: str = "#3498db") -> str:
    """Return HTML for a styled KPI metric card."""
    delta_html = f'<p style="font-size:12px;color:#7f8c8d;margin:0">{delta}</p>' if delta else ""
    return f"""
    <div style="background:{color};border-radius:8px;padding:16px 20px;color:white;text-align:center">
      <p style="font-size:13px;margin:0;opacity:0.85">{label}</p>
      <p style="font-size:28px;font-weight:bold;margin:4px 0">{value}</p>
      {delta_html}
    </div>
    """


def segment_color(segment_name: str) -> str:
    return SEGMENT_COLORS.get(segment_name, "#95a5a6")


def risk_color(risk_level: str) -> str:
    return RISK_COLORS.get(str(risk_level).lower(), "#95a5a6")


def fmt_currency(value: float) -> str:
    if value >= 1_000_000:
        return f"${value/1_000_000:.1f}M"
    if value >= 1_000:
        return f"${value/1_000:.1f}K"
    return f"${value:.0f}"
