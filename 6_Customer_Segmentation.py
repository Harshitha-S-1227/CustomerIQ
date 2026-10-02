"""
pages/6_🧩_Customer_Segmentation.py — Group and understand customers.

Deliberately different from Batch Prediction: Batch Prediction's job is
"predict many customers"; this page's job is "group and understand
customers who are already predicted." It is a PURE CONSUMER of the
canonical processed-results cache (services.session_manager) — it does
NOT upload a CSV or call /copilot/batch itself. If Batch Prediction
hasn't been run yet this session, this page tells the user to run it
first rather than silently reprocessing the same file (see
sm.has_processed_results()).
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from components.charts import build_bar_chart, build_donut_chart, build_segment_scatter
from components.notifications import render_empty_state
from components.sidebar import render_sidebar
from constants import SEGMENT_ICONS
from services import session_manager as sm
from utils.formatting import format_currency, format_pct

st.set_page_config(page_title="Segmentation — CustomerIQ", page_icon="🧩", layout="wide")
sm.init_session_state()
render_sidebar()

st.title("🧩 Customer Segmentation")
st.caption("Group and understand the customers already scored by Batch Prediction.")

# ---------------------------------------------------------------------------
# Data source — pure consumer of the canonical processed-results cache
# ---------------------------------------------------------------------------

if not sm.has_processed_results():
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    go_to_batch = render_empty_state(
        "No processed customers yet this session. Please run Batch "
        "Prediction first — Segmentation reuses those results rather "
        "than re-uploading and re-scoring the same file.",
        icon="🧩",
        action_label="Go to Batch Prediction →",
    )
    if go_to_batch:
        st.switch_page("pages/3_📁_Batch_Prediction.py")
    st.stop()

seg_df: pd.DataFrame = sm.get("processed_results_df")
n_customers = len(seg_df)

st.caption(
    f"Showing {n_customers:,} customers from **{sm.get('processed_source_filename')}** "
    f"(processed {sm.get('processed_at_utc')} UTC)."
)
st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Segment distribution — count AND percentage per segment
# ---------------------------------------------------------------------------

col1, col2 = st.columns([1, 1])
with col1:
    with st.container(border=True):
        seg_counts = seg_df["segment"].value_counts()
        st.plotly_chart(
            build_donut_chart(
                labels=seg_counts.index.tolist(),
                values=seg_counts.values.tolist(),
                title="Segment Distribution",
            ),
            width="stretch",
        )
with col2:
    with st.container(border=True):
        avg_clv_by_seg = seg_df.groupby("segment")["predicted_clv_brl"].mean().sort_values()
        st.plotly_chart(
            build_bar_chart(
                x=avg_clv_by_seg.index.tolist(),
                y=avg_clv_by_seg.values.tolist(),
                title="Avg. CLV by Segment",
                orientation="h",
                x_title="Avg. CLV (₹)",
            ),
            width="stretch",
        )

with st.container(border=True):
    st.markdown("**Segment Characteristics**")
    seg_summary = (
        seg_df.groupby("segment")
        .agg(
            customers=("customer_id", "count"),
            avg_clv_brl=("predicted_clv_brl", "mean"),
            avg_churn_score=("churn_risk_score", "mean"),
            high_churn_pct=("churn_risk", lambda s: (s == "High").mean()),
        )
        .reset_index()
        .sort_values("customers", ascending=False)
    )
    seg_summary["pct_of_cohort"] = seg_summary["customers"] / n_customers
    display_summary = pd.DataFrame(
        {
            "Segment": seg_summary["segment"],
            "Customers": seg_summary["customers"],
            "% of Cohort": seg_summary["pct_of_cohort"].apply(format_pct),
            "Avg. CLV": seg_summary["avg_clv_brl"].apply(format_currency),
            "Avg. Churn Score": seg_summary["avg_churn_score"].apply(lambda v: f"{v:.2f}"),
            "% High Churn Risk": seg_summary["high_churn_pct"].apply(format_pct),
        }
    )
    st.dataframe(display_summary, hide_index=True, width="stretch")

# ---------------------------------------------------------------------------
# Interactive scatter — CLV vs priority, colored by segment
# ---------------------------------------------------------------------------

with st.container(border=True):
    st.markdown("**CLV vs. Priority Score, by Segment**")
    st.plotly_chart(
        build_segment_scatter(
            seg_df,
            x="predicted_clv_brl",
            y="priority_score",
            color_col="segment",
            hover_cols=["customer_id", "churn_risk"],
        ),
        width="stretch",
    )

# ---------------------------------------------------------------------------
# Filters + table + drill-down
# ---------------------------------------------------------------------------

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
st.markdown("### Customer Drill-Down")

filt_col1, filt_col2, filt_col3 = st.columns(3)
with filt_col1:
    segment_filter = st.multiselect("Segment", options=sorted(seg_df["segment"].unique()))
with filt_col2:
    churn_filter = st.multiselect("Churn Risk", options=sorted(seg_df["churn_risk"].unique()))
with filt_col3:
    search_id = st.text_input("Search customer ID")

filtered = seg_df.copy()
if segment_filter:
    filtered = filtered[filtered["segment"].isin(segment_filter)]
if churn_filter:
    filtered = filtered[filtered["churn_risk"].isin(churn_filter)]
if search_id:
    filtered = filtered[filtered["customer_id"].astype(str).str.contains(search_id, case=False, na=False)]

st.caption(f"{len(filtered):,} of {n_customers:,} customers match the current filters.")

display_df = filtered.copy()
display_df["predicted_clv_brl"] = display_df["predicted_clv_brl"].apply(format_currency)
display_df["total_opportunity_brl"] = display_df["total_opportunity_brl"].apply(format_currency)
display_df["priority_score"] = display_df["priority_score"].apply(lambda v: f"{v:.3f}")

st.dataframe(
    display_df.rename(columns={
        "customer_id": "Customer ID", "segment": "Segment", "clv_tier": "Tier",
        "churn_risk": "Churn Risk", "churn_risk_score": "Churn Score",
        "predicted_clv_brl": "CLV", "priority_score": "Priority",
        "total_opportunity_brl": "Opportunity",
    }),
    hide_index=True,
    width="stretch",
    height=380,
)

selected_id = st.selectbox(
    "Select a customer to inspect", options=["—"] + filtered["customer_id"].astype(str).tolist()
)
if selected_id != "—":
    row = filtered[filtered["customer_id"].astype(str) == selected_id].iloc[0]
    with st.container(border=True):
        icon = SEGMENT_ICONS.get(row["segment"], "🏷️")
        st.markdown(f"#### {icon} {selected_id}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Segment", row["segment"])
        c2.metric("CLV", format_currency(row["predicted_clv_brl"]))
        c3.metric("Churn Risk", row["churn_risk"], f"{row['churn_risk_score']:.2f}")
        c4.metric("Priority", f"{row['priority_score']:.3f}")
        st.caption(
            "For full driver-level detail (why this customer scored this "
            "way), open Explainability and select this customer."
        )
