"""
pages/5_🧠_Explainability.py — Why a customer's CLV was predicted this way.

Two tabs:
  - "Single Customer" -> calls both copilot_single (CLV, tier, segment,
    churn risk — /explain alone doesn't return these) and explain_single
    (base_value, prediction_log_scale, precise SHAP drivers) for the same
    customer. The PRIMARY view is business language: actual feature
    values plus a qualitative impact label (Strong/Moderate/Slight
    Positive/Negative), never raw SHAP numbers or log-scale jargon up
    front. The technical SHAP chart, waterfall, and log-scale figures are
    NOT removed — they live under an "Advanced model details" expander
    for anyone who wants them.
  - "Cohort / Global" -> POST /explain/batch -> global feature
    importance for a batch of customers (mean |SHAP| per feature,
    population-level — not per-row). Reuses Batch Prediction's cached
    raw feature rows when available instead of requiring a re-upload.

Explainability is deliberately different from Batch Prediction: Batch
Prediction scores MANY customers fast (no SHAP by default); this page
explains WHY one specific customer scored the way they did (always with
real SHAP, since that's the whole point here) — see app.py's
_run_lightweight_copilot docstring for why those are different code
paths.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from components.charts import build_feature_importance_chart, build_shap_driver_chart, build_waterfall_chart
from components.customer_form import render_customer_form
from components.feature_value_table import build_driver_display_df, render_business_drivers, render_feature_values_table
from components.loaders import spinner
from components.notifications import render_api_error, render_empty_state, success
from components.sidebar import render_sidebar
from services import session_manager as sm
from services.api_client import ApiError, copilot_single, explain_batch, explain_single, get_model_info
from utils.formatting import format_currency

st.set_page_config(page_title="Explainability — CustomerIQ", page_icon="🧠", layout="wide")
sm.init_session_state()
render_sidebar()

st.title("🧠 Explainability")
st.caption("Why a customer's CLV was predicted the way it was — in business language, with technical detail available on request.")

if sm.get("model_info") is None:
    try:
        with spinner("Loading model schema..."):
            sm.set_value("model_info", get_model_info())
    except ApiError as exc:
        render_api_error(exc, context="Loading model schema")

model_info = sm.get("model_info") or {}
known_categories = model_info.get("known_categories", {})

tab_single, tab_global = st.tabs(["🔍 Single Customer", "🌐 Cohort / Global Importance"])

# ---------------------------------------------------------------------------
# Tab 1: Single-customer explanation
# ---------------------------------------------------------------------------

with tab_single:
    with st.container(border=True):
        form_values = render_customer_form(
            known_categories=known_categories, key_prefix="explain_page", submit_label="Explain Prediction"
        )

    if form_values is not None:
        try:
            with spinner("Computing prediction and SHAP values..."):
                # Two calls, both fast (proven — see app.py's async->def fix
                # and the lightweight-endpoint timing test): copilot_single
                # gives the business-facing header (CLV, tier, segment, churn
                # risk — none of which /explain alone returns), explain_single
                # gives the precise base_value/prediction_log_scale figures
                # for the Advanced model details section below. Neither call
                # reprocesses any dataset — both operate on this one customer.
                copilot_result = copilot_single(form_values)
                expl = explain_single(form_values)
            sm.update(expl_single_result=expl, expl_single_copilot=copilot_result)
            success(f"Explained in {expl['computation_time_sec']:.3f}s.")
        except ApiError as exc:
            render_api_error(exc, context="Computing SHAP explanation")

    expl = sm.get("expl_single_result")
    copilot_result = sm.get("expl_single_copilot")
    if expl and copilot_result:
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        # --- Business-facing header: CLV, tier, segment, churn risk ---
        cid = copilot_result.get("customer_id") or expl.get("customer_id") or "n/a"
        st.markdown(f"#### Customer: `{cid}`")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Predicted CLV", format_currency(copilot_result["predicted_clv"]))
        m2.metric("CLV Tier", copilot_result["clv_tier"])
        m3.metric("Segment", copilot_result["segment"])
        m4.metric("Churn Risk", copilot_result["churn_risk"])

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # --- Primary explanation: business language, actual feature values ---
        render_business_drivers(expl["top_positive_drivers"], expl["top_negative_drivers"])

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        render_feature_values_table(
            expl["top_positive_drivers"], expl["top_negative_drivers"]
        )

        # --- Advanced model details: SHAP chart, waterfall, log-scale figures ---
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        with st.expander("⚙️ Advanced model details"):
            st.caption(
                "Technical detail for anyone who wants it — SHAP values on "
                "the model's internal log1p scale, not something to read as "
                "currency."
            )
            am1, am2, am3 = st.columns(3)
            am1.metric("Base Value (log scale)", f"{expl['base_value']:.4f}")
            am2.metric("Prediction (log scale)", f"{expl['prediction_log_scale']:.4f}")
            am3.metric("Model", expl["model_version"])

            col_l, col_r = st.columns(2)
            with col_l:
                st.markdown("**SHAP Feature Drivers**")
                st.plotly_chart(
                    build_shap_driver_chart(
                        expl["top_positive_drivers"], expl["top_negative_drivers"]
                    ),
                    width="stretch",
                )
            with col_r:
                st.markdown("**Prediction Waterfall**")
                st.plotly_chart(
                    build_waterfall_chart(
                        base_value=expl["base_value"],
                        positive_drivers=expl["top_positive_drivers"],
                        negative_drivers=expl["top_negative_drivers"],
                        final_value=expl["prediction_log_scale"],
                    ),
                    width="stretch",
                )

            col_pos, col_neg = st.columns(2)
            with col_pos:
                st.markdown("**Top Positive Drivers (raw SHAP)**")
                pos_df = build_driver_display_df(expl["top_positive_drivers"])
                if not pos_df.empty:
                    st.dataframe(pos_df, hide_index=True, width="stretch")
            with col_neg:
                st.markdown("**Top Negative Drivers (raw SHAP)**")
                neg_df = build_driver_display_df(expl["top_negative_drivers"])
                if not neg_df.empty:
                    st.dataframe(neg_df, hide_index=True, width="stretch")

        st.caption(f"customer_id: `{cid}`")
    else:
        render_empty_state(
            "Submit the form above to see why this customer's CLV was predicted.",
            icon="🔍",
        )

# ---------------------------------------------------------------------------
# Tab 2: Cohort / global importance
# ---------------------------------------------------------------------------

with tab_global:
    st.write(
        "Computes mean absolute SHAP value per feature across a batch of "
        "customers — a population-level view of what drives CLV, via "
        "POST /explain/batch."
    )

    raw_df = sm.get("processed_raw_features_df")
    source_hint = (
        f"Uses the {len(raw_df)} customers currently loaded from "
        f"{sm.get('processed_source_filename')} (processed "
        f"{sm.get('processed_at_utc') or 'recently'})."
        if raw_df is not None and not raw_df.empty
        else "No cohort loaded yet — run a Batch Prediction first, or upload a CSV below."
    )
    st.caption(source_hint)

    uploaded = st.file_uploader("Or upload a CSV of customers to explain", type=["csv"], key="expl_batch_upload")

    if raw_df is not None and not raw_df.empty and uploaded is None:
        st.caption(
            f"⏱️ Computing SHAP for all {len(raw_df):,} customers can take "
            "a while at the high end of the row range — this is a real "
            "SHAP pass, not something to run repeatedly."
        )

    run_global = st.button("Compute Global Importance", type="primary")

    if run_global:
        customers_payload: list[dict] | None = None
        if uploaded is not None:
            try:
                df = pd.read_csv(uploaded)
                customers_payload = df.where(pd.notnull(df), None).to_dict(orient="records")
            except Exception as exc:  # noqa: BLE001 — surfaced to user, not swallowed
                render_api_error(
                    ApiError(message=f"Could not parse CSV: {exc}"), context="Reading uploaded file"
                )
        elif raw_df is not None and not raw_df.empty:
            # processed_raw_features_df is the actual uploaded CustomerFeatures
            # rows from Batch Prediction (unlike the old segmentation_df, which
            # only held derived summary columns) — reusable as-is, no re-upload
            # needed, matching the "reuse processed results" architecture.
            customers_payload = raw_df.where(pd.notnull(raw_df), None).to_dict(orient="records")

        if customers_payload:
            try:
                with spinner(f"Computing SHAP values for {len(customers_payload)} customers..."):
                    batch_expl = explain_batch(customers_payload)
                sm.set_value("expl_batch_result", batch_expl)
                success(f"Computed in {batch_expl['computation_time_sec']:.2f}s.")
            except ApiError as exc:
                render_api_error(exc, context="Computing global importance")

    batch_expl = sm.get("expl_batch_result")
    if batch_expl:
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        m1.metric("Customers", batch_expl["n_customers"])
        m2.metric("Base Value", f"{batch_expl['base_value']:.4f}")
        m3.metric("Compute Time", f"{batch_expl['computation_time_sec']:.2f}s")

        importance_df = pd.DataFrame(batch_expl["global_feature_importance"])
        with st.container(border=True):
            st.plotly_chart(
                build_feature_importance_chart(importance_df),
                width="stretch",
            )
        st.dataframe(importance_df, hide_index=True, width="stretch")
