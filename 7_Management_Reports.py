"""
pages/7_📑_Management_Reports.py — Executive reports from ALREADY processed results.

Pure consumer of the canonical processed-results cache (see
services.session_manager): the report's core numbers — CLV summary,
segment distribution, churn risk distribution, campaign prioritisation,
top revenue opportunities, model health — are all computed CLIENT-SIDE
from `processed_results_df` / `processed_insights`, produced once by
Batch Prediction. No re-upload, no re-running prediction, no
re-running segmentation.

The one thing that genuinely can't be reconstructed from cached data is
**global SHAP feature importance** — Batch Prediction's default fast
path deliberately skips SHAP entirely (see app.py's
_run_lightweight_copilot), so there's no SHAP output sitting in cache to
reuse. That single piece is offered as an explicit, opt-in action here
(POST /explain/batch on the already-uploaded feature rows) rather than
silently re-running the whole batch pipeline.

Exports: CSV and Excel are built client-side directly from cached data
(no backend round-trip needed since nothing new was computed). PDF
reuses utils.pdf_export unchanged.
"""

from __future__ import annotations

import io
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from components.charts import build_bar_chart, build_donut_chart, build_feature_importance_chart
from components.loaders import spinner
from components.notifications import render_api_error, render_empty_state, success
from components.sidebar import render_sidebar
from services import session_manager as sm
from services.api_client import ApiError, explain_batch
from utils.formatting import format_currency, format_pct

st.set_page_config(page_title="Management Reports — CustomerIQ", page_icon="📑", layout="wide")
sm.init_session_state()
render_sidebar()

st.title("📑 Management Reports")
st.caption("Executive summary built from the customers Batch Prediction already scored.")

# ---------------------------------------------------------------------------
# Data source — pure consumer, same gate as Segmentation/Dashboard
# ---------------------------------------------------------------------------

if not sm.has_processed_results():
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    go_to_batch = render_empty_state(
        "No processed customers yet this session. Please run Batch "
        "Prediction first — Management Reports reuses those results "
        "instead of re-uploading and re-scoring the same file.",
        icon="📑",
        action_label="Go to Batch Prediction →",
    )
    if go_to_batch:
        st.switch_page("pages/3_📁_Batch_Prediction.py")
    st.stop()

results_df: pd.DataFrame = sm.get("processed_results_df")
insights: list[dict] = sm.get("processed_insights") or []
n_customers = len(results_df)


def _build_report_from_cache(results_df: pd.DataFrame, insights: list[dict]) -> dict:
    """Compute the same report shape POST /management-report returns,
    entirely from already-cached Batch Prediction output — no backend call."""
    clv = results_df["predicted_clv_brl"]
    clv_summary = {
        "total_predicted_revenue_brl": round(float(clv.sum()), 2),
        "total_opportunity_brl": round(float(results_df["total_opportunity_brl"].sum()), 2),
        "mean_clv_brl": round(float(clv.mean()), 2),
        "median_clv_brl": round(float(clv.median()), 2),
        "p75_clv_brl": round(float(clv.quantile(0.75)), 2),
    }

    campaign_counts: dict[str, int] = {}
    model_versions: set[str] = set()
    confidences: list[float] = []
    for ins in insights:
        actions = ins.get("recommended_actions") or []
        primary = actions[0] if actions else "No Action Required"
        campaign_counts[primary] = campaign_counts.get(primary, 0) + 1
        if ins.get("model_version"):
            model_versions.add(ins["model_version"])
        if ins.get("confidence_score") is not None:
            confidences.append(ins["confidence_score"])

    top_opp = (
        results_df.sort_values("total_opportunity_brl", ascending=False)
        .head(10)[["customer_id", "predicted_clv_brl", "total_opportunity_brl", "segment"]]
        .to_dict(orient="records")
    )

    mean_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    n_low_confidence = sum(1 for c in confidences if c < 0.4)

    return {
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "n_customers": n_customers,
        "clv_summary": clv_summary,
        "segment_distribution": results_df["segment"].value_counts().to_dict(),
        "churn_risk_distribution": results_df["churn_risk"].value_counts().to_dict(),
        "campaign_prioritisation": campaign_counts,
        "top_revenue_opportunities": top_opp,
        "model_health": {
            "model_versions": sorted(model_versions) or ["unknown"],
            "mean_confidence": mean_confidence,
            "n_low_confidence": n_low_confidence,
        },
        "top_global_features": sm.get("report_result").get("top_global_features")
        if sm.get("report_result") else None,
    }


report = _build_report_from_cache(results_df, insights)

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
st.caption(
    f"Source: **{sm.get('processed_source_filename')}** · "
    f"{n_customers:,} customers · processed {sm.get('processed_at_utc')} UTC"
)

# ---------------------------------------------------------------------------
# Optional: on-demand global SHAP importance
# ---------------------------------------------------------------------------

with st.expander("📊 Global SHAP Feature Importance (optional, computed on demand)"):
    st.write(
        "Batch Prediction's default fast path skips SHAP entirely, so "
        "there's no driver-level detail cached to reuse for this. "
        "Computing it explicitly here runs one additional POST "
        "/explain/batch call over the cached feature rows — a genuine, "
        "separate SHAP pass, not a silent re-run of batch prediction."
    )
    if st.button("Compute Global Feature Importance"):
        raw_df = sm.get("processed_raw_features_df")
        if raw_df is None or raw_df.empty:
            st.warning("Raw feature rows aren't available for this cohort.")
        else:
            try:
                payload = raw_df.where(pd.notnull(raw_df), None).to_dict(orient="records")
                with spinner(f"Computing SHAP importance for {len(payload)} customers..."):
                    batch_expl = explain_batch(payload)
                sm.set_value("report_result", {"top_global_features": batch_expl["global_feature_importance"]})
                success("Global feature importance computed.")
                st.rerun()
            except ApiError as exc:
                render_api_error(exc, context="Computing global feature importance")

# ---------------------------------------------------------------------------
# Report preview
# ---------------------------------------------------------------------------

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
st.markdown("### Report Preview")

clv = report["clv_summary"]
m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Predicted Revenue", format_currency(clv["total_predicted_revenue_brl"], compact=True))
m2.metric("Total Opportunity", format_currency(clv["total_opportunity_brl"], compact=True))
m3.metric("Mean CLV", format_currency(clv["mean_clv_brl"]))
m4.metric("P75 CLV", format_currency(clv["p75_clv_brl"]))

col1, col2, col3 = st.columns(3)
with col1:
    with st.container(border=True):
        dist = report["segment_distribution"]
        st.plotly_chart(
            build_donut_chart(list(dist.keys()), list(dist.values()), title="Segments"),
            width="stretch",
        )
with col2:
    with st.container(border=True):
        dist = report["churn_risk_distribution"]
        st.plotly_chart(
            build_donut_chart(list(dist.keys()), list(dist.values()), title="Churn Risk"),
            width="stretch",
        )
with col3:
    with st.container(border=True):
        dist = report["campaign_prioritisation"]
        st.plotly_chart(
            build_bar_chart(
                x=list(dist.keys()), y=list(dist.values()),
                title="Primary Campaign Assignments", orientation="h",
            ),
            width="stretch",
        )

with st.container(border=True):
    st.markdown("**Top 10 Revenue Opportunities**")
    top_opp = pd.DataFrame(report["top_revenue_opportunities"])
    if not top_opp.empty:
        top_opp_display = top_opp.copy()
        for col in ("predicted_clv_brl", "total_opportunity_brl"):
            top_opp_display[col] = top_opp_display[col].apply(format_currency)
        st.dataframe(top_opp_display, hide_index=True, width="stretch")

if report.get("top_global_features"):
    with st.container(border=True):
        st.markdown("**Top Global SHAP Features**")
        gi_df = pd.DataFrame(report["top_global_features"])
        st.plotly_chart(
            build_feature_importance_chart(gi_df, value_col="mean_abs_shap", title=""),
            width="stretch",
        )

with st.container(border=True):
    st.markdown("**Model Health**")
    health = report["model_health"]
    hc1, hc2, hc3 = st.columns(3)
    hc1.metric("Model Version(s)", ", ".join(health["model_versions"]))
    hc2.metric("Mean Confidence", f"{health['mean_confidence']:.0%}")
    hc3.metric("Low-Confidence Customers", health["n_low_confidence"])

# ---------------------------------------------------------------------------
# Downloads — built client-side from cached data; no new backend call needed
# ---------------------------------------------------------------------------

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
st.markdown("### Downloads")
st.caption("Built directly from the cached results above — CSV/Excel need no backend call; PDF is rendered in this app.")

dl1, dl2, dl3 = st.columns(3)

with dl1:
    csv_bytes = results_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV", data=csv_bytes,
        file_name="customeriq_management_report.csv",
        mime="text/csv", width="stretch",
    )

with dl2:
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        results_df.to_excel(writer, sheet_name="Customer Results", index=False)
        pd.DataFrame(report["top_revenue_opportunities"]).to_excel(
            writer, sheet_name="Top Opportunities", index=False
        )
    st.download_button(
        "Download Excel", data=excel_buffer.getvalue(),
        file_name="customeriq_management_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        width="stretch",
    )

with dl3:
    try:
        from utils.pdf_export import build_management_report_pdf

        pdf_bytes = build_management_report_pdf(report)
        st.download_button(
            "Download PDF", data=pdf_bytes,
            file_name="customeriq_management_report.pdf",
            mime="application/pdf", width="stretch",
        )
    except Exception as exc:  # noqa: BLE001
        st.button("PDF generation failed", disabled=True, width="stretch")
        st.caption(f"⚠️ {exc}")

with st.expander("View raw report data"):
    st.json(report)
