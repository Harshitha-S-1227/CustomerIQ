"""
pages/4_🤖_AI_Copilot.py — AI Copilot, chat-styled.

Honest framing: the backend has no free-text/LLM chat endpoint — /copilot
and /copilot/batch return one structured AIInsight for a specific
customer's feature payload (see ai_copilot.py). This page presents that
capability as a conversation: the user selects/enters a customer, taps
a suggested prompt, and the assistant "replies" by calling /copilot and
rendering the relevant slice of the real response as a chat bubble.
Nothing here is generated free-text from an LLM — every reply is
templated from actual CopilotResponse fields.
"""

from __future__ import annotations

import streamlit as st

from components.charts import build_shap_driver_chart
from components.customer_form import render_customer_form
from components.feature_value_table import render_feature_values_table
from components.loaders import spinner
from components.notifications import render_api_error
from components.sidebar import render_sidebar
from constants import CAMPAIGN_ICONS, SEGMENT_ICONS, SUGGESTED_PROMPTS
from services import session_manager as sm
from services.api_client import ApiError, copilot_single, get_model_info
from utils.formatting import format_currency, sanitize_currency_text

st.set_page_config(page_title="AI Copilot — CustomerIQ", page_icon="🤖", layout="wide")
sm.init_session_state()
render_sidebar()

st.title("🤖 AI Copilot")
st.caption(
    "Ask about a customer and get real AI-driven insight — every reply "
    "here comes from a live POST /copilot call, not free-text generation."
)

# ---------------------------------------------------------------------------
# Customer selection — a compact form in a popover keeps the chat area clean.
# ---------------------------------------------------------------------------

if sm.get("model_info") is None:
    try:
        with spinner("Loading model schema..."):
            sm.set_value("model_info", get_model_info())
    except ApiError as exc:
        render_api_error(exc, context="Loading model schema")

model_info = sm.get("model_info") or {}
known_categories = model_info.get("known_categories", {})

active_customer = sm.get("chat_active_customer")

top_bar = st.columns([4, 1])
with top_bar[0]:
    if active_customer:
        cid = active_customer.get("customer_unique_id") or "unnamed customer"
        st.markdown(f"**Active customer:** `{cid}` — {len([k for k in active_customer if active_customer[k] is not None])} features set")
    else:
        st.info("No customer selected yet. Set one below to start the conversation.", icon="👤")
with top_bar[1]:
    if st.button("Reset chat", width="stretch"):
        sm.reset_chat()
        st.rerun()

with st.popover("👤 Set / change active customer", width="stretch"):
    form_values = render_customer_form(
        known_categories=known_categories, key_prefix="copilot_page", submit_label="Set Customer"
    )
    if form_values is not None:
        sm.update(chat_active_customer=form_values)
        sm.append_chat_message(
            "user", "I'd like to talk about this customer."
        )
        sm.append_chat_message(
            "assistant",
            "Got it — customer set. Ask me something below, or tap a "
            "suggested prompt.",
        )
        st.rerun()

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Suggested prompts
# ---------------------------------------------------------------------------

st.markdown("**Suggested prompts**")
prompt_cols = st.columns(len(SUGGESTED_PROMPTS))
clicked_action: str | None = None
for col, prompt in zip(prompt_cols, SUGGESTED_PROMPTS):
    with col:
        if st.button(prompt["label"], width="stretch", disabled=active_customer is None):
            clicked_action = prompt["action"]
            sm.append_chat_message("user", prompt["label"])

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)


def _run_copilot_and_reply(action: str) -> None:
    """Call /copilot for the active customer and append a templated reply."""
    try:
        with spinner("Analyzing customer..."):
            result = copilot_single(active_customer)
    except ApiError as exc:
        sm.append_chat_message(
            "assistant", f"⚠️ I couldn't reach the backend: {exc.message}"
        )
        return

    # business_summary/nl_explanation are backend-assembled prose that may
    # embed a literal "R$" (see ai_copilot.py's summary generator) —
    # sanitize once here rather than at every content-building branch below.
    result["business_summary"] = sanitize_currency_text(result["business_summary"])
    result["nl_explanation"] = sanitize_currency_text(result["nl_explanation"])

    if action == "analyze_single":
        content = (
            f"**{result['segment']} {SEGMENT_ICONS.get(result['segment'], '')}** · "
            f"CLV **{format_currency(result['predicted_clv'])}** ({result['clv_tier']} tier) · "
            f"Churn risk **{result['churn_risk']}**.\n\n{result['business_summary']}"
        )
    elif action == "explain_drivers":
        content = result["nl_explanation"]
    elif action == "recommend_campaign":
        actions = result["recommended_actions"]
        icons_list = ", ".join(f"{CAMPAIGN_ICONS.get(a, '')} {a}" for a in actions) or "No action required"
        content = f"Recommended actions, in priority order: {icons_list}"
    elif action == "churn_summary":
        sentences = [s.strip() for s in result["nl_explanation"].split(".") if s.strip()]
        risk_sentence = f" {sentences[-1]}." if sentences else ""
        content = (
            f"Churn risk is **{result['churn_risk']}** "
            f"(score {result['churn_risk_score']:.2f}).{risk_sentence}"
        )
    else:
        content = result["business_summary"]

    sm.append_chat_message("assistant", content, data=result)


if clicked_action and active_customer:
    _run_copilot_and_reply(clicked_action)
    st.rerun()

# ---------------------------------------------------------------------------
# Free-text box — mapped to the closest supported action, framed honestly.
# ---------------------------------------------------------------------------

user_text = st.chat_input(
    "Type a question (mapped to the closest supported analysis)...",
    disabled=active_customer is None,
)
if user_text:
    sm.append_chat_message("user", user_text)
    lowered = user_text.lower()
    if any(w in lowered for w in ["driver", "why", "explain"]):
        action = "explain_drivers"
    elif any(w in lowered for w in ["campaign", "action", "recommend"]):
        action = "recommend_campaign"
    elif any(w in lowered for w in ["churn", "risk", "leave"]):
        action = "churn_summary"
    else:
        action = "analyze_single"
    _run_copilot_and_reply(action)
    st.rerun()

# ---------------------------------------------------------------------------
# Chat transcript
# ---------------------------------------------------------------------------

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
history = sm.get("chat_history")

if not history:
    st.caption("No messages yet — set a customer and tap a suggested prompt to begin.")

for msg_idx, msg in enumerate(history):
    with st.chat_message(msg["role"], avatar="🤖" if msg["role"] == "assistant" else "🧑"):
        st.markdown(msg["content"])
        data = msg.get("data")
        if data and msg["role"] == "assistant":
            with st.expander("View SHAP drivers for this reply"):
                st.plotly_chart(
                    build_shap_driver_chart(
                        data["top_positive_drivers"], data["top_negative_drivers"]
                        ),
                        width="stretch",
                        key=f"copilot_shap_chart_{msg_idx}",
                        )
