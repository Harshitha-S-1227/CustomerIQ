"""
config.py — Central configuration for the CustomerIQ Streamlit frontend.

Single source of truth for:
  - API connection settings (base URL, timeouts, retry policy)
  - Page metadata (title, icon, layout)
  - Theme tokens (colors, typography) used by components/charts.py
    and injected as CSS by streamlit_app.py

Environment variables override defaults so the same codebase can point
at local dev, staging, or production backends without code changes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# API configuration
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.getenv("CUSTOMERIQ_API_URL", "http://localhost:8000")

# Separate timeouts because the full-SHAP batch endpoints (/copilot/batch,
# /explain/batch, /management-report with global importance) still run
# TreeExplainer over the whole request and can legitimately take over a
# minute at MAX_BATCH_ROWS (measured ~6s per 500 rows -> ~125s at 10,000
# rows) — this is why the Batch Prediction page's default flow instead
# calls /copilot/batch/lightweight (no per-row SHAP, same 10,000 rows in
# well under a second) rather than raising this timeout further.
API_TIMEOUT_DEFAULT: float = float(os.getenv("CUSTOMERIQ_API_TIMEOUT", "15"))
API_TIMEOUT_BATCH: float = float(os.getenv("CUSTOMERIQ_API_TIMEOUT_BATCH", "180"))
API_TIMEOUT_UPLOAD: float = float(os.getenv("CUSTOMERIQ_API_TIMEOUT_UPLOAD", "180"))

API_MAX_RETRIES: int = int(os.getenv("CUSTOMERIQ_API_MAX_RETRIES", "2"))
API_RETRY_BACKOFF_SEC: float = 0.75

# Mirrors the backend's own limit (src/data/constants.py: MAX_ROWS, used
# by app.py's BatchCustomerFeatures/ManagementReportRequest max_length and
# the /upload-csv row check) so the frontend can fail fast with a friendly
# message instead of waiting on a 413/422 round trip.
#
# CURRENT DEVELOPMENT LIMIT: 10,000 rows — kept in lockstep with the
# backend's MAX_ROWS. This bounds validation; it isn't a performance
# target on its own. The real reason batch operations stay fast at this
# size is architectural: Batch Prediction scores customers via
# /copilot/batch/lightweight (no per-row SHAP — see api_client.py), and
# every other page (Dashboard, Segmentation, Management Reports) reuses
# those cached results instead of re-uploading and reprocessing the same
# CSV. Raising this later is a one-line change here plus the matching
# one-line change in the backend's constants.py — nothing else in either
# codebase hard-codes a row count.
MAX_BATCH_ROWS: int = 10_000
MAX_UPLOAD_ROWS: int = 10_000

# Sized for the current 10,000-row limit (a CSV that size is a few MB at
# most for this schema) rather than a much larger cap sized for a
# 1,000,000-row limit — no reason to allow a huge upload against a
# 10,000-row validation ceiling.
MAX_UPLOAD_FILE_MB: int = 50

# ---------------------------------------------------------------------------
# Page metadata
# ---------------------------------------------------------------------------

APP_TITLE: str = "CustomerIQ"
APP_TAGLINE: str = "Customer Lifetime Value Intelligence Platform"
APP_ICON: str = "📊"
PAGE_LAYOUT: str = "wide"
INITIAL_SIDEBAR_STATE: str = "expanded"

# ---------------------------------------------------------------------------
# Theme tokens
# ---------------------------------------------------------------------------
# Enterprise SaaS palette (Power BI / Salesforce / Azure Portal register).
# Every component and chart builder pulls colors from here rather than
# hardcoding hex values, so the whole app can be re-themed from one place.


@dataclass(frozen=True)
class Theme:
    """Attribute *names* are stable across theme revisions — every
    component/page references THEME.<name>, so this dataclass's default
    values are the only thing that needs to change to re-skin the app."""

    # Brand
    primary: str = "#2563EB"          # Primary Blue
    primary_dark: str = "#1D4ED8"     # Primary Hover
    primary_light: str = "#60A5FA"
    primary_bg: str = "#EFF6FF"       # blue-50 — chip/badge fills

    # Neutrals
    bg: str = "#F8FAFC"
    surface: str = "#FFFFFF"
    surface_alt: str = "#F1F5F9"
    border: str = "#E5E7EB"
    text_primary: str = "#111827"
    text_secondary: str = "#6B7280"
    text_muted: str = "#9CA3AF"

    # Semantic
    success: str = "#10B981"
    success_bg: str = "#ECFDF5"
    warning: str = "#F59E0B"
    warning_bg: str = "#FFFBEB"
    danger: str = "#EF4444"
    danger_bg: str = "#FEF2F2"
    info: str = "#2563EB"
    info_bg: str = "#EFF6FF"

    # High-value / premium accent (VIP segment, top-tier badges)
    purple: str = "#7C3AED"
    purple_bg: str = "#F5F3FF"

    # Sidebar — dark, distinct from the light card surfaces (Azure Portal /
    # Power BI style left nav)
    sidebar_bg: str = "#0F172A"
    sidebar_text: str = "#F8FAFC"
    sidebar_text_muted: str = "#94A3B8"
    sidebar_active: str = "#2563EB"
    sidebar_hover: str = "#1E293B"

    # Chart palette — used in order for categorical series.
    chart_categorical: tuple[str, ...] = (
        "#2563EB",  # primary blue
        "#10B981",  # success green
        "#F59E0B",  # warning orange
        "#7C3AED",  # purple
        "#EF4444",  # danger red
        "#0EA5E9",  # sky
        "#6B7280",  # neutral grey
        "#DB2777",  # pink
    )

    # Typography
    font_family: str = (
        "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', "
        "Roboto, Helvetica, Arial, sans-serif"
    )

    # Layout — 16px rounded corners on primary surfaces.
    radius_sm: str = "8px"
    radius_md: str = "12px"
    radius_lg: str = "16px"
    shadow_sm: str = "0 1px 2px 0 rgba(17, 24, 39, 0.05)"
    shadow_md: str = "0 4px 16px -4px rgba(17, 24, 39, 0.10)"

    def risk_colors(self) -> dict[str, str]:
        """Semantic color per churn_risk value (High/Medium/Low)."""
        return {"High": self.danger, "Medium": self.warning, "Low": self.success}

    def tier_colors(self) -> dict[str, str]:
        """Semantic color per clv_tier value (High/Medium/Low)."""
        return {"High": self.primary, "Medium": self.info, "Low": self.text_muted}


THEME = Theme()

# ---------------------------------------------------------------------------
# Misc app constants
# ---------------------------------------------------------------------------

CURRENCY_SYMBOL: str = "₹"
DATE_FORMAT: str = "%d %b %Y, %H:%M UTC"
SAMPLE_CSV_MAX_PREVIEW_ROWS: int = 20
