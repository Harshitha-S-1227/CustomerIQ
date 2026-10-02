"""
pages/3_📁_Batch_Prediction.py — Score MANY customers at once.

"Batch Prediction" means predicting many customers — this is the ONE
place in the app a CSV gets uploaded and processed. Every other page
(Dashboard, Customer Segmentation, Management Reports) reuses the
results this page produces instead of independently re-uploading and
re-scoring the same file — see services.session_manager's
`processed_*` keys and `has_processed_results()`.

Default flow calls POST /copilot/batch/lightweight (prediction +
segmentation + churn risk, NO per-row SHAP) rather than the full-SHAP
/copilot/batch — measured ~13x faster at scale, and segment/churn-risk
are numerically identical either way since neither depends on SHAP
(see app.py's _run_lightweight_copilot docstring). Explaining WHY a
specific customer scored the way they did is Explainability's job
(pages/5), not Batch Prediction's — that page calls /explain or
/copilot for one customer at a time, which stays fast regardless.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from components.loaders import spinner
from components.notifications import error, render_api_error, success, warning
from components.sidebar import render_sidebar
from components.kpi_card import render_kpi_row
from config import MAX_UPLOAD_FILE_MB, MAX_UPLOAD_ROWS, THEME
from constants import FEATURE_FIELDS
from services import session_manager as sm
from services.api_client import ApiError, copilot_batch, copilot_batch_lightweight
from utils.formatting import format_currency, format_number, format_pct
from utils.validators import is_blocking_problem, validate_batch_dataframe, validate_upload_file

st.set_page_config(page_title="Batch Prediction — CustomerIQ", page_icon="📁", layout="wide")
sm.init_session_state()
render_sidebar()

st.title("📁 Batch Prediction")
st.caption(
    f"Predict MANY customers at once — upload a CSV to score up to "
    f"{MAX_UPLOAD_ROWS:,} customers in one request."
)

with st.expander("Expected CSV schema", expanded=False):
    st.write(
        "The file should include columns matching the model's feature "
        "schema. Any recognised columns are used; unrecognised columns "
        "are ignored. Missing values are handled natively by the model."
    )
    schema_df = pd.DataFrame(
        [{"column": f["name"], "type": f["type"], "group": f["group"]} for f in FEATURE_FIELDS]
    )
    st.dataframe(schema_df, hide_index=True, width="stretch", height=240)

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    f"Upload customer CSV (max {MAX_UPLOAD_FILE_MB:,} MB, up to {MAX_UPLOAD_ROWS:,} customers)",
    type=["csv"],
)

include_shap = st.checkbox(
    "Also compute full SHAP drivers for every customer (slower — not needed "
    "for the results below; only useful if you plan to export per-row "
    "driver detail). Leave unchecked for normal use.",
    value=False,
)

col_run, col_clear = st.columns([1, 1])
run_clicked = col_run.button(
    "Run Batch Prediction", type="primary", width="stretch", disabled=uploaded_file is None
)
if col_clear.button("Clear Results", width="stretch"):
    sm.clear_processed_results()
    st.rerun()

if run_clicked and uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    file_problems = validate_upload_file(uploaded_file.name, file_bytes)
    blocking = [p for p in file_problems if is_blocking_problem(p)]

    if blocking:
        for p in blocking:
            error(p)
    else:
        try:
            df_in = pd.read_csv(uploaded_file)
        except Exception as exc:  # noqa: BLE001
            df_in = None
            error(f"Could not parse CSV: {exc}")

        if df_in is not None:
            df_problems = validate_batch_dataframe(df_in)
            blocking2 = [p for p in df_problems if is_blocking_problem(p)]
            for p in df_problems:
                if not is_blocking_problem(p):
                    warning(p)

            if blocking2:
                for p in blocking2:
                    error(p)
            else:
                payload = df_in.where(pd.notnull(df_in), None).to_dict(orient="records")
                try:
                    label = (
                        "Scoring customers with full SHAP drivers (this takes "
                        "longer)..." if include_shap else
                        "Scoring customers (prediction + segmentation + churn "
                        "risk)..."
                    )
                    with spinner(label):
                        insights = (
                            copilot_batch(payload) if include_shap
                            else copilot_batch_lightweight(payload)
                        )

                    rows = [
                        {
                            "customer_id": ins["customer_id"],
                            "predicted_clv_brl": ins["predicted_clv"],
                            "clv_tier": ins["clv_tier"],
                            "segment": ins["segment"],
                            "churn_risk": ins["churn_risk"],
                            "churn_risk_score": ins["churn_risk_score"],
                            "priority_score": ins["revenue_opportunity"]["priority_score"],
                            "total_opportunity_brl": ins["revenue_opportunity"]["total_opportunity_brl"],
                        }
                        for ins in insights
                    ]
                    results_df = pd.DataFrame(rows)

                    sm.update(
                        processed_results_df=results_df,
                        processed_raw_features_df=df_in,
                        processed_insights=insights,
                        processed_source_filename=uploaded_file.name,
                        processed_at_utc=datetime.now(tz=timezone.utc).isoformat(),
                        processed_shap_included=include_shap,
                    )
                    success(f"Scored {len(results_df):,} customers from '{uploaded_file.name}'.")
                except ApiError as exc:
                    render_api_error(exc, context="Batch prediction")

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

if not sm.has_processed_results():
    st.info("Upload a CSV and click **Run Batch Prediction** to see results here.", icon="👆")
    st.stop()

results_df: pd.DataFrame = sm.get("processed_results_df")
source_name = sm.get("processed_source_filename")

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
st.markdown("### Results")
st.caption(
    f"Source: **{source_name}** · processed {sm.get('processed_at_utc')} UTC"
    + (" · includes full SHAP drivers" if sm.get("processed_shap_included") else "")
)

n_customers = len(results_df)
avg_clv = float(results_df["predicted_clv_brl"].mean())
tier_counts = results_df["clv_tier"].value_counts()
n_high = int(tier_counts.get("High", 0))
n_medium = int(tier_counts.get("Medium", 0))
n_low = int(tier_counts.get("Low", 0))
n_at_risk = int((results_df["churn_risk"] == "High").sum())

render_kpi_row(
    [
        dict(label="Customers Analysed", value=format_number(n_customers), icon="👥", accent=THEME.primary),
        dict(label="Average Predicted CLV", value=format_currency(avg_clv), icon="💰", accent=THEME.success),
        dict(label="High-Value Customers", value=format_number(n_high), icon="⭐", accent=THEME.purple),
        dict(label="At-Risk Customers", value=format_number(n_at_risk), icon="⚠️", accent=THEME.danger),
    ]
)

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

col_tier, col_churn = st.columns(2)
with col_tier:
    with st.container(border=True):
        st.markdown("**CLV Tier Breakdown**")
        tier_df = pd.DataFrame(
            [
                {"Tier": "High", "Customers": n_high, "% of Cohort": format_pct(n_high / n_customers)},
                {"Tier": "Medium", "Customers": n_medium, "% of Cohort": format_pct(n_medium / n_customers)},
                {"Tier": "Low", "Customers": n_low, "% of Cohort": format_pct(n_low / n_customers)},
            ]
        )
        st.dataframe(tier_df, hide_index=True, width="stretch")

with col_churn:
    with st.container(border=True):
        st.markdown("**Churn Risk Distribution**")
        churn_counts = results_df["churn_risk"].value_counts()
        churn_df = pd.DataFrame(
            [
                {"Churn Risk": risk, "Customers": int(n), "% of Cohort": format_pct(n / n_customers)}
                for risk, n in churn_counts.items()
            ]
        )
        st.dataframe(churn_df, hide_index=True, width="stretch")

with st.container(border=True):
    st.markdown("**Customer Results**")
    display_df = results_df[
        ["customer_id", "predicted_clv_brl", "clv_tier", "segment", "churn_risk"]
    ].copy()
    display_df["predicted_clv_brl"] = display_df["predicted_clv_brl"].apply(format_currency)
    st.dataframe(
        display_df.rename(
            columns={
                "customer_id": "Customer ID",
                "predicted_clv_brl": "Predicted CLV",
                "clv_tier": "CLV Tier",
                "segment": "Segment",
                "churn_risk": "Churn Risk",
            }
        ),
        hide_index=True,
        width="stretch",
        height=420,
    )

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
st.markdown("### Download Results")
st.caption(
    "Built directly from the results above — no extra backend round-trip."
)
csv_bytes = results_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download Results (CSV)",
    data=csv_bytes,
    file_name=f"customeriq_batch_results_{n_customers}.csv",
    mime="text/csv",
    width="stretch",
)

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
st.info(
    "**Segmentation**, **Dashboard**, and **Management Reports** now reuse "
    "these results automatically — no need to upload this file again.",
    icon="🔗",
)
