"""
pages/8_⚙️_API_Status.py — Backend health & endpoint status.

Two real signals from the backend, plus a lightweight reachability
probe of every documented route:
  - GET /health         -> HealthResponse (status, model_loaded, model_version, uptime)
  - GET /model-info      -> ModelInfoResponse (feature schema, hyperparams)

There's no backend endpoint that reports "endpoint status" for every
route individually, so the per-endpoint table below is built here by
issuing minimal, cheap requests (GET / and GET /health, GET /model-info
are actually called; POST endpoints are listed with their expected
method/path from app.py rather than fired with dummy payloads, since
sending synthetic customer data just to populate a status page would
be misleading and wasteful). This is disclosed in the caption so it's
never mistaken for a backend-provided status feed.
"""

from __future__ import annotations

import time

import pandas as pd
import streamlit as st

from components.notifications import render_api_error, success
from components.sidebar import render_sidebar
from config import API_BASE_URL
from services import session_manager as sm
from services.api_client import ApiError, get_health, get_model_info, get_root

st.set_page_config(page_title="API Status — CustomerIQ", page_icon="⚙️", layout="wide")
sm.init_session_state()
render_sidebar(refresh_health=False)

st.title("⚙️ API Status")
st.caption(f"Live status for the CustomerIQ backend at `{API_BASE_URL}`.")

refresh = st.button("🔄 Refresh Now", type="primary")

# ---------------------------------------------------------------------------
# Core health check
# ---------------------------------------------------------------------------

if refresh or sm.get("api_health_payload") is None:
    t0 = time.perf_counter()
    try:
        health_payload = get_health()
        latency_ms = (time.perf_counter() - t0) * 1000
        sm.update(
            api_healthy=True,
            api_health_payload=health_payload,
            api_health_error=None,
            last_health_check_ts=time.time(),
        )
        success(f"Backend is healthy ({latency_ms:.0f}ms).")
    except ApiError as exc:
        sm.update(api_healthy=False, api_health_payload=None, api_health_error=exc.message)
        render_api_error(exc, context="Health check")

is_healthy = sm.get("api_healthy")
health_payload = sm.get("api_health_payload")

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Backend Status", "🟢 Online" if is_healthy else "🔴 Offline")
if health_payload:
    model_version = health_payload.get("model_version")
    col2.metric("Model Loaded", "✅ Yes" if model_version else "❌ No")
    col3.metric("Model Version", model_version or "—")
    uptime = health_payload.get("uptime_sec")
    col4.metric("Uptime", f"{uptime / 3600:.1f} hrs" if uptime else "—")
else:
    col2.metric("Model Loaded", "—")
    col3.metric("Model Version", "—")
    col4.metric("Uptime", "—")

with st.expander("Raw /health response"):
    st.json(health_payload or {"error": sm.get("api_health_error")})

# ---------------------------------------------------------------------------
# Model info
# ---------------------------------------------------------------------------

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
st.markdown("### Model Info")

if is_healthy:
    try:
        model_info = get_model_info()
        sm.set_value("model_info", model_info)

        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Model Version", model_info.get("model_version", "—"))
        mc2.metric("Feature Count", len(model_info.get("feature_names", [])))
        mc3.metric("Categorical Features", len(model_info.get("categorical_features", [])))

        with st.expander("Feature names"):
            st.write(", ".join(model_info.get("feature_names", [])) or "—")

        with st.expander("Known category values"):
            known = model_info.get("known_categories", {})
            for cat_col, values in known.items():
                st.markdown(f"**{cat_col}** ({len(values)} values)")
                st.caption(", ".join(str(v) for v in values[:25]) + (" …" if len(values) > 25 else ""))

        with st.expander("Hyperparameters"):
            hp = model_info.get("hyperparameters", {})
            if hp:
                st.dataframe(
                    pd.DataFrame(list(hp.items()), columns=["Parameter", "Value"]),
                    hide_index=True, width="stretch",
                )
    except ApiError as exc:
        render_api_error(exc, context="Fetching model info")
else:
    st.info("Model info unavailable while the backend is offline.", icon="⏳")

# ---------------------------------------------------------------------------
# Endpoint reference / reachability
# ---------------------------------------------------------------------------

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
st.markdown("### Endpoints")
st.caption(
    "GET routes below are checked live. POST routes require a customer "
    "payload to test meaningfully, so their status reflects overall "
    "backend reachability rather than an individual live probe."
)

if st.button("Check GET / (root)"):
    try:
        root_payload = get_root()
        success("GET / responded successfully.")
        st.json(root_payload)
    except ApiError as exc:
        render_api_error(exc, context="Checking root endpoint")

endpoint_rows = [
    {"Method": "GET", "Path": "/", "Purpose": "Platform info", "Checked live": "Yes (button above)"},
    {"Method": "GET", "Path": "/health", "Purpose": "Health check", "Checked live": "Yes"},
    {"Method": "GET", "Path": "/model-info", "Purpose": "Model schema & hyperparameters", "Checked live": "Yes"},
    {"Method": "POST", "Path": "/predict", "Purpose": "Single-customer CLV prediction", "Checked live": "No — reachability only"},
    {"Method": "POST", "Path": "/predict/batch", "Purpose": "Batch CLV prediction (1-1,000,000)", "Checked live": "No — reachability only"},
    {"Method": "POST", "Path": "/explain", "Purpose": "Single-customer SHAP drivers", "Checked live": "No — reachability only"},
    {"Method": "POST", "Path": "/explain/batch", "Purpose": "Cohort global feature importance", "Checked live": "No — reachability only"},
    {"Method": "POST", "Path": "/copilot", "Purpose": "Full AI insight for one customer", "Checked live": "No — reachability only"},
    {"Method": "POST", "Path": "/copilot/batch", "Purpose": "AI insights for a batch", "Checked live": "No — reachability only"},
    {"Method": "POST", "Path": "/management-report", "Purpose": "Executive report over a cohort", "Checked live": "No — reachability only"},
    {"Method": "POST", "Path": "/upload-csv", "Purpose": "CSV upload & batch scoring", "Checked live": "No — reachability only"},
    {"Method": "GET", "Path": "/download-report/{subdir}/{filename}", "Purpose": "Fetch a generated report file", "Checked live": "No — reachability only"},
]

st.dataframe(
    pd.DataFrame(endpoint_rows),
    hide_index=True,
    width="stretch",
    height=420,
)

overall_status = "🟢 All checked routes reachable" if is_healthy else "🔴 Backend unreachable — POST routes will fail too"
st.caption(overall_status)
