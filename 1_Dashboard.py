"""
pages/1_📊_Dashboard.py — Portfolio overview.

Pure consumer of the canonical processed-results cache (see
services.session_manager): reads `processed_results_df` /
`processed_raw_features_df`, produced once by Batch Prediction. Does
NOT independently re-upload or re-score anything — if nothing has been
processed yet this session, it points the user at Batch Prediction
rather than fabricating numbers.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from components.charts import build_bar_chart, build_donut_chart
from components.kpi_card import KpiDelta, render_kpi_row
from components.notifications import render_empty_state
from components.sidebar import render_sidebar
from config import THEME
from services import session_manager as sm
from utils.formatting import format_currency, format_feature_value, format_number, format_pct, get_feature_label

st.set_page_config(page_title="Dashboard — CustomerIQ", page_icon="📊", layout="wide")
sm.init_session_state()
render_sidebar()

st.title("📊 Dashboard")
st.caption("Portfolio overview for the customers Batch Prediction has scored this session.")

# ---------------------------------------------------------------------------
# Resolve the active cohort — canonical processed-results cache only.
# ---------------------------------------------------------------------------

if not sm.has_processed_results():
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    go_to_batch = render_empty_state(
        "No cohort scored yet this session. Run a batch prediction to "
        "populate the dashboard.",
        icon="📭",
        action_label="Go to Batch Prediction →",
    )
    if go_to_batch:
        st.switch_page("pages/3_📁_Batch_Prediction.py")
    st.stop()

seg_df: pd.DataFrame = sm.get("processed_results_df")

# ---------------------------------------------------------------------------
# KPI row — Customers Analysed / Average CLV / High-Value Customers /
# Revenue Opportunity (matches the home page's hero KPIs for consistency).
# ---------------------------------------------------------------------------

n_customers = len(seg_df)
mean_clv = float(seg_df["predicted_clv_brl"].mean())
n_high_value = int((seg_df["clv_tier"] == "High").sum())
total_opportunity = float(seg_df["total_opportunity_brl"].sum())
high_risk_pct = float((seg_df["churn_risk"] == "High").mean())

render_kpi_row(
    [
        dict(
            label="Customers Analysed",
            value=format_number(n_customers),
            icon="👥",
            accent=THEME.primary,
        ),
        dict(
            label="Average CLV",
            value=format_currency(mean_clv),
            icon="💰",
            accent=THEME.success,
        ),
        dict(
            label="High-Value Customers",
            value=format_number(n_high_value),
            icon="⭐",
            accent=THEME.purple,
            help_text=f"{format_pct(n_high_value / n_customers)} of cohort",
        ),
        dict(
            label="Revenue Opportunity",
            value=format_currency(total_opportunity, compact=True),
            icon="🚀",
            accent=THEME.info,
            help_text="Upsell + retention + cross-sell value",
        ),
    ]
)

st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Segment + churn + tier distribution charts
# ---------------------------------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    with st.container(border=True):
        seg_counts = seg_df["segment"].value_counts()
        st.plotly_chart(
            build_donut_chart(
                labels=seg_counts.index.tolist(),
                values=seg_counts.values.tolist(),
                title="Customer Segments",
                color_map=None,
            ),
            width="stretch",
        )

with col2:
    with st.container(border=True):
        churn_counts = seg_df["churn_risk"].value_counts()
        st.plotly_chart(
            build_donut_chart(
                labels=churn_counts.index.tolist(),
                values=churn_counts.values.tolist(),
                title="Churn Risk",
                color_map=THEME.risk_colors(),
            ),
            width="stretch",
        )
        if high_risk_pct > 0.25:
            st.caption(f"⚠️ {format_pct(high_risk_pct)} of this cohort is high churn-risk.")

with col3:
    with st.container(border=True):
        tier_counts = seg_df["clv_tier"].value_counts()
        st.plotly_chart(
            build_donut_chart(
                labels=tier_counts.index.tolist(),
                values=tier_counts.values.tolist(),
                title="CLV Tier",
                color_map=THEME.tier_colors(),
            ),
            width="stretch",
        )

# ---------------------------------------------------------------------------
# Revenue by segment
# ---------------------------------------------------------------------------

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
col_a, col_b = st.columns([3, 2])

with col_a:
    with st.container(border=True):
        rev_by_segment = (
            seg_df.groupby("segment")["predicted_clv_brl"].sum().sort_values(ascending=True)
        )
        st.plotly_chart(
            build_bar_chart(
                x=rev_by_segment.index.tolist(),
                y=rev_by_segment.values.tolist(),
                title="Predicted Revenue by Segment",
                orientation="h",
                color_map=None,
                x_title="Revenue (₹)",
            ),
            width="stretch",
        )

with col_b:
    with st.container(border=True):
        st.markdown("**Top 10 Priority Customers**")
        top10 = seg_df.sort_values("priority_score", ascending=False).head(10)[
            ["customer_id", "segment", "predicted_clv_brl", "priority_score"]
        ].copy()
        top10["predicted_clv_brl"] = top10["predicted_clv_brl"].apply(format_currency)
        top10["priority_score"] = top10["priority_score"].apply(lambda v: f"{v:.2f}")
        st.dataframe(
            top10.rename(
                columns={
                    "customer_id": "Customer",
                    "segment": "Segment",
                    "predicted_clv_brl": "CLV",
                    "priority_score": "Priority",
                }
            ),
            hide_index=True,
            width="stretch",
            height=380,
        )

# ---------------------------------------------------------------------------
# Customer Profile Snapshot — real averaged feature values (actual days,
# item counts, currency) rather than vague tier labels like "High" /
# "Medium" / "Low". Sourced from the raw uploaded CustomerFeatures rows
# cached by Batch Prediction — there is no backend endpoint that returns
# cohort-wide raw feature averages, so this is computed client-side from
# data the user already submitted.
# ---------------------------------------------------------------------------

raw_df: pd.DataFrame | None = sm.get("processed_raw_features_df")

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
st.markdown("### Customer Profile Snapshot")

if raw_df is None or raw_df.empty:
    st.caption("Real customer-value averages appear here once Batch Prediction has run.")
else:
    st.caption("Average values across the scored cohort — actual customer data, not tier labels.")

    # (field name, icon) — every one of these genuinely exists in the
    # CustomerFeatures schema (app.py) and constants.FEATURE_FIELDS.
    # "avg_order_value" is deliberately NOT included here: it isn't a real
    # input field (it's engineered/target-adjacent and dropped before
    # training — see src/models/model_training.py's prepare_features_target
    # drop list), so showing it would mean fabricating a number.
    _snapshot_fields = [
        ("tenure_days", "⏳"),
        ("recency_days", "🕒"),
        ("total_items_purchased", "🛍️"),
        ("avg_shipping_days", "🚚"),
        ("avg_installments", "💳"),
        ("avg_freight_value", "📦"),
    ]

    snapshot_cards = []
    for field_name, icon in _snapshot_fields:
        if field_name not in raw_df.columns:
            continue
        series = raw_df[field_name].dropna()
        if series.empty:
            continue
        snapshot_cards.append(
            dict(
                label=get_feature_label(field_name),
                value=format_feature_value(field_name, float(series.mean())),
                icon=icon,
                accent=THEME.primary,
                help_text=f"Avg across {len(series):,} customers",
            )
        )

    if not snapshot_cards:
        st.caption("None of the expected feature columns were found in the uploaded file.")
    else:
        for row_start in range(0, len(snapshot_cards), 3):
            render_kpi_row(snapshot_cards[row_start : row_start + 3])
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Data source footnote
# ---------------------------------------------------------------------------

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
st.caption(
    f"Showing {n_customers:,} customers from **{sm.get('processed_source_filename')}** "
    f"(processed {sm.get('processed_at_utc')} UTC). Re-run Batch Prediction to refresh."
)
