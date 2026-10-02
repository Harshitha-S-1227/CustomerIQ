"""
pages/2_🔮_Customer_Prediction.py — Single-customer scoring.

Flow:
  1. Render the CustomerFeatures form (components.customer_form).
  2. On submit: POST /copilot (single call gets prediction + SHAP +
     segmentation + churn + recommendations in one response — cheaper
     than calling /predict, /explain, and /copilot separately for the
     same customer).
  3. Render CLV, tier, segment, churn risk, confidence, and business
     insights from the CopilotResponse.
"""

from __future__ import annotations

import streamlit as st

from components.charts import build_gauge_chart, build_shap_driver_chart
from components.customer_form import render_customer_form, render_customer_form_summary
from components.feature_value_table import render_feature_values_table
from components.kpi_card import render_kpi_row
from components.loaders import spinner
from components.notifications import render_api_error, success
from components.sidebar import render_sidebar
from config import THEME
from constants import CAMPAIGN_ICONS, SEGMENT_ICONS
from services import session_manager as sm
from services.api_client import ApiError, copilot_single, get_model_info
from utils.formatting import format_currency, sanitize_currency_text

st.set_page_config(page_title="Customer Prediction — CustomerIQ", page_icon="🔮", layout="wide")
sm.init_session_state()
render_sidebar()

st.title("🔮 Customer Prediction")
st.caption(
    "Score a single customer's 90-day CLV, segment, churn risk, and "
    "recommended actions."
)

# Load known categories once (for the top_category dropdown) — cached in
# session so we don't hit /model-info on every rerun of this page.
if sm.get("model_info") is None:
    try:
        with spinner("Loading model schema..."):
            sm.set_value("model_info", get_model_info())
    except ApiError as exc:
        render_api_error(exc, context="Loading model schema")

model_info = sm.get("model_info") or {}
known_categories = model_info.get("known_categories", {})

st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

with st.container(border=True):
    form_values = render_customer_form(
        known_categories=known_categories, key_prefix="predict_page", submit_label="Predict CLV"
    )

if form_values is not None:
    sm.update(cp_form_values=form_values)
    sm.clear_prediction_results()
    try:
        with spinner("Running prediction, SHAP explanation, and AI analysis..."):
            result = copilot_single(form_values)
        # business_summary is backend-assembled prose that may embed a
        # literal "R$" (see ai_copilot.py's summary generator) — sanitize
        # once here rather than at every render site below.
        result["business_summary"] = sanitize_currency_text(result["business_summary"])
        result["nl_explanation"] = sanitize_currency_text(result["nl_explanation"])
        sm.update(cp_copilot=result, cp_last_customer_id=result.get("customer_id"))
        success("Prediction complete.")
    except ApiError as exc:
        render_api_error(exc, context="Generating prediction")

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

result = sm.get("cp_copilot")
if not result:
    st.info("Fill in the form above and click **Predict CLV** to see results here.", icon="👆")
    st.stop()

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
st.markdown("### Results")

render_customer_form_summary(sm.get("cp_form_values") or {})

render_kpi_row(
    [
        dict(
            label="Predicted CLV (90-day)",
            value=format_currency(result["predicted_clv"]),
            icon="💰",
            accent=THEME.primary,
        ),
        dict(
            label="CLV Tier",
            value=result["clv_tier"],
            icon="🏆",
            accent=THEME.tier_colors().get(result["clv_tier"], THEME.primary),
        ),
        dict(
            label="Segment",
            value=result["segment"],
            icon=SEGMENT_ICONS.get(result["segment"], "🏷️"),
            accent=THEME.primary,
        ),
        dict(
            label="Churn Risk",
            value=result["churn_risk"],
            icon="⚠️" if result["churn_risk"] == "High" else "✅",
            accent=THEME.risk_colors().get(result["churn_risk"], THEME.text_muted),
            help_text=f"Score: {result['churn_risk_score']:.2f}",
        ),
    ]
)

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

col_left, col_right = st.columns([3, 2])

with col_left:
    with st.container(border=True):
        st.markdown("**Business Summary**")
        st.write(result["business_summary"])
        st.markdown("**Explanation**")
        st.write(result["nl_explanation"])

    with st.container(border=True):
        st.markdown("**SHAP Drivers**")
        st.plotly_chart(
            build_shap_driver_chart(
                result["top_positive_drivers"], result["top_negative_drivers"]
            ),
            width="stretch",
        )

    render_feature_values_table(
        result["top_positive_drivers"], result["top_negative_drivers"]
    )

with col_right:
    with st.container(border=True):
        st.markdown("**Confidence Score**")
        st.plotly_chart(
            build_gauge_chart(result["confidence_score"], title=""),
            width="stretch",
        )
        st.caption(f"Model: `{result['model_version']}`")

    with st.container(border=True):
        st.markdown("**Recommended Actions**")
        for action in result["recommended_actions"]:
            icon = CAMPAIGN_ICONS.get(action, "▶️")
            st.markdown(f"{icon} {action}")

    ro = result["revenue_opportunity"]
    with st.container(border=True):
        st.markdown("**Revenue Opportunity**")
        st.markdown(f"Upsell: **{format_currency(ro['upsell_opportunity_brl'])}**")
        st.markdown(f"Retention value: **{format_currency(ro['retention_value_brl'])}**")
        st.markdown(f"Cross-sell lift: **{format_currency(ro['cross_sell_lift_brl'])}**")
        st.markdown(f"Total opportunity: **{format_currency(ro['total_opportunity_brl'])}**")
        st.progress(min(ro["priority_score"], 1.0), text=f"Priority score: {ro['priority_score']:.2f}")

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
with st.expander("View raw response"):
    st.json(result)
