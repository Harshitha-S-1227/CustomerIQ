"""
constants.py — Static values mirrored from the CustomerIQ backend.

These are copied deliberately (not imported) because the frontend is a
separate deployable that only talks to the backend over HTTP — it has
no access to the `src.*` Python package. Every value below is sourced
from a specific backend file so it can be re-verified against the API
at any time:

  - CustomerSegment, ChurnRisk, CampaignType   -> src/models/ai_copilot.py
  - VALID_PAYMENT_TYPES, BRAZIL_STATES,
    VALID_ORDER_STATUSES                       -> src/data/constants.py
  - CustomerFeatures field list & constraints  -> app.py (Pydantic model)

Categorical values that the *model* actually learned (e.g. the exact
set of `top_category` strings) are NOT hardcoded here — they differ by
training run. The frontend fetches those live from GET /model-info
(`known_categories`) and caches them in session_state. Anything in
this file is stable regardless of which model version is loaded.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# AI Copilot enums (src/models/ai_copilot.py)
# ---------------------------------------------------------------------------

CUSTOMER_SEGMENTS: list[str] = [
    "VIP",
    "High Value",
    "Loyal",
    "Growing",
    "New",
    "At Risk",
    "Dormant",
    "Low Value",
]

# Emoji glyphs used purely for scannability in tables/cards — never the
# sole indicator of meaning (color + text label always present too).
SEGMENT_ICONS: dict[str, str] = {
    "VIP": "💎",
    "High Value": "⭐",
    "Loyal": "🤝",
    "Growing": "📈",
    "New": "✨",
    "At Risk": "⚠️",
    "Dormant": "💤",
    "Low Value": "🔻",
}

CHURN_RISK_LEVELS: list[str] = ["High", "Medium", "Low"]

CAMPAIGN_TYPES: list[str] = [
    "Retention Campaign",
    "Cross-Sell",
    "Upsell",
    "Loyalty Program",
    "Premium Membership",
    "Personalized Discount",
    "Win-Back Campaign",
    "Referral Rewards",
    "No Action Required",
]

CAMPAIGN_ICONS: dict[str, str] = {
    "Retention Campaign": "🛡️",
    "Cross-Sell": "🔀",
    "Upsell": "⬆️",
    "Loyalty Program": "🏆",
    "Premium Membership": "👑",
    "Personalized Discount": "🏷️",
    "Win-Back Campaign": "🔁",
    "Referral Rewards": "🎁",
    "No Action Required": "✅",
}

CLV_TIERS: list[str] = ["High", "Medium", "Low"]

EXPORT_FORMATS: list[str] = ["csv", "excel", "json"]

# ---------------------------------------------------------------------------
# Domain constants (src/data/constants.py)
# ---------------------------------------------------------------------------

VALID_PAYMENT_TYPES: list[str] = [
    "credit_card",
    "boleto",
    "voucher",
    "debit_card",
    "not_defined",
]

PAYMENT_TYPE_LABELS: dict[str, str] = {
    "credit_card": "Credit Card",
    "boleto": "Boleto",
    "voucher": "Voucher",
    "debit_card": "Debit Card",
    "not_defined": "Not Defined",
}

BRAZIL_STATES: list[str] = [
    "AC", "AL", "AP", "AM", "BA", "CE", "DF", "ES", "GO",
    "MA", "MT", "MS", "MG", "PA", "PB", "PR", "PE", "PI",
    "RJ", "RN", "RS", "RO", "RR", "SC", "SP", "SE", "TO",
]

# Full state names for display — standard Brazilian state names (ISO
# 3166-2:BR), not derived from the model; used purely to make
# customer_state human-readable in tables (e.g. "SP" -> "São Paulo").
STATE_NAMES: dict[str, str] = {
    "AC": "Acre", "AL": "Alagoas", "AP": "Amapá", "AM": "Amazonas",
    "BA": "Bahia", "CE": "Ceará", "DF": "Distrito Federal",
    "ES": "Espírito Santo", "GO": "Goiás", "MA": "Maranhão",
    "MT": "Mato Grosso", "MS": "Mato Grosso do Sul", "MG": "Minas Gerais",
    "PA": "Pará", "PB": "Paraíba", "PR": "Paraná", "PE": "Pernambuco",
    "PI": "Piauí", "RJ": "Rio de Janeiro", "RN": "Rio Grande do Norte",
    "RS": "Rio Grande do Sul", "RO": "Rondônia", "RR": "Roraima",
    "SC": "Santa Catarina", "SP": "São Paulo", "SE": "Sergipe",
    "TO": "Tocantins",
}

# ---------------------------------------------------------------------------
# Region — Indian states & Union Territories
# ---------------------------------------------------------------------------
# IMPORTANT HONESTY NOTE (see the final summary for the full explanation):
# the trained model's `region` categorical vocabulary is the 5 Brazilian
# macro-regions it learned from the Olist training data (North, Northeast,
# Midwest, Southeast, South) — see known_categories['region'] from
# GET /model-info. These Indian values are NOT mapped back to a Brazilian
# region under the hood (that would be exactly the fabrication this list
# exists to avoid). Sent as-is, "Karnataka" is simply outside the model's
# trained vocabulary; the backend's existing preprocessing
# (preprocess_prediction_input in src/models/prediction.py) already
# replaces any out-of-vocabulary categorical value with NaN before
# prediction — a pre-existing, safe fallback, not something added for
# this change — so the model treats it as "region unknown" rather than
# crashing or silently receiving a fabricated Brazilian label. In
# practice this means `region` currently contributes no signal to
# predictions made with an Indian state selected. Genuinely modeling
# Indian regions would require retraining on Indian-labeled data.
REGIONS: list[str] = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
    "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand",
    "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur",
    "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab", "Rajasthan",
    "Sikkim", "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh",
    "Uttarakhand", "West Bengal",
    # Union Territories
    "Andaman and Nicobar Islands", "Chandigarh",
    "Dadra and Nagar Haveli and Daman and Diu", "Delhi",
    "Jammu and Kashmir", "Ladakh", "Lakshadweep", "Puducherry",
]

# ---------------------------------------------------------------------------
# CustomerFeatures form schema (app.py — Pydantic model field-by-field)
# ---------------------------------------------------------------------------
# Drives components/customer_form.py. `group` controls which form section
# a field renders in. Constraints (ge/le) mirror the Pydantic Field(...)
# validators exactly so client-side validation never falls out of sync
# with what the API will 422 on.

FEATURE_FIELDS: list[dict] = [
    # --- Identity -----------------------------------------------------
    {"name": "customer_unique_id", "label": "Customer ID", "type": "text",
     "group": "identity", "required": False,
     "help": "Optional identifier, not used by the model for prediction."},

    # --- Recency / Frequency ------------------------------------------
    {"name": "recency_days", "label": "Recency (days since last order)",
     "type": "number", "group": "activity", "min": 0.0, "step": 1.0, "default": 45.0},
    {"name": "frequency", "label": "Frequency (delivered orders)",
     "type": "int", "group": "activity", "min": 1, "step": 1, "default": 2},
    {"name": "total_orders", "label": "Total Orders", "type": "int",
     "group": "activity", "min": 1, "step": 1, "default": 2},
    {"name": "delivered_orders", "label": "Delivered Orders", "type": "int",
     "group": "activity", "min": 0, "step": 1, "default": 2},
    {"name": "cancelled_orders", "label": "Cancelled Orders", "type": "int",
     "group": "activity", "min": 0, "step": 1, "default": 0},
    {"name": "order_success_rate", "label": "Order Success Rate", "type": "float",
     "group": "activity", "min": 0.0, "max": 1.0, "step": 0.01, "default": 1.0,
     "format": "pct"},
    {"name": "customer_lifetime_days", "label": "Customer Lifetime (days)",
     "type": "int", "group": "activity", "min": 0, "step": 1, "default": 120},
    {"name": "tenure_days", "label": "Tenure (days)", "type": "int",
     "group": "activity", "min": 0, "step": 1, "default": 120},

    # --- Payment --------------------------------------------------------
    {"name": "avg_installments", "label": "Avg. Installments", "type": "float",
     "group": "payment", "min": 1.0, "step": 0.1, "default": 2.0},
    {"name": "pct_credit_card_payments", "label": "% Credit Card Payments",
     "type": "float", "group": "payment", "min": 0.0, "max": 1.0, "step": 0.01,
     "default": 0.7, "format": "pct"},
    {"name": "most_common_payment_type", "label": "Most Common Payment Type",
     "type": "select", "group": "payment", "options": VALID_PAYMENT_TYPES,
     "option_labels": PAYMENT_TYPE_LABELS, "default": "credit_card"},

    # --- Reviews ----------------------------------------------------------
    {"name": "avg_review_score", "label": "Avg. Review Score", "type": "float",
     "group": "reviews", "min": 1.0, "max": 5.0, "step": 0.1, "default": 4.2},
    {"name": "min_review_score", "label": "Min Review Score", "type": "float",
     "group": "reviews", "min": 1.0, "max": 5.0, "step": 0.1, "default": 3.0},
    {"name": "max_review_score", "label": "Max Review Score", "type": "float",
     "group": "reviews", "min": 1.0, "max": 5.0, "step": 0.1, "default": 5.0},
    {"name": "review_count", "label": "Review Count", "type": "float",
     "group": "reviews", "min": 0.0, "step": 1.0, "default": 2.0},
    {"name": "pct_low_reviews", "label": "% Low Reviews (1–2 stars)", "type": "float",
     "group": "reviews", "min": 0.0, "max": 1.0, "step": 0.01, "default": 0.0,
     "format": "pct"},
    {"name": "pct_five_star_reviews", "label": "% Five-Star Reviews", "type": "float",
     "group": "reviews", "min": 0.0, "max": 1.0, "step": 0.01, "default": 0.6,
     "format": "pct"},

    # --- Delivery -----------------------------------------------------
    {"name": "avg_delivery_days", "label": "Avg. Delivery Days", "type": "float",
     "group": "delivery", "min": 0.0, "step": 0.5, "default": 12.0},
    {"name": "avg_shipping_days", "label": "Avg. Shipping Days", "type": "float",
     "group": "delivery", "min": 0.0, "step": 0.5, "default": 8.0},
    {"name": "late_delivery_pct", "label": "% Late Deliveries", "type": "float",
     "group": "delivery", "min": 0.0, "max": 1.0, "step": 0.01, "default": 0.1,
     "format": "pct"},
    {"name": "early_delivery_pct", "label": "% Early Deliveries", "type": "float",
     "group": "delivery", "min": 0.0, "max": 1.0, "step": 0.01, "default": 0.6,
     "format": "pct"},
    {"name": "n_delivered_with_dates", "label": "Delivered Orders w/ Dates",
     "type": "int", "group": "delivery", "min": 0, "step": 1, "default": 2},

    # --- Basket ---------------------------------------------------------
    {"name": "avg_items_per_order", "label": "Avg. Items / Order", "type": "float",
     "group": "basket", "min": 0.0, "step": 0.1, "default": 1.3},
    {"name": "avg_freight_value", "label": "Avg. Freight Value (₹)", "type": "float",
     "group": "basket", "min": 0.0, "step": 1.0, "default": 18.0},
    {"name": "total_items_purchased", "label": "Total Items Purchased",
     "type": "int", "group": "basket", "min": 0, "step": 1, "default": 3},
    {"name": "unique_categories", "label": "Unique Categories", "type": "int",
     "group": "basket", "min": 1, "step": 1, "default": 2},
    {"name": "category_hhi", "label": "Category Concentration (HHI)", "type": "float",
     "group": "basket", "min": 0.0, "max": 1.0, "step": 0.01, "default": 0.6,
     "help": "Herfindahl-Hirschman index of category concentration; 1.0 = all "
             "purchases in one category."},
    {"name": "top_category", "label": "Top Category", "type": "select_dynamic",
     "group": "basket", "default": None,
     "help": "Populated from the live model's known categories (GET /model-info)."},

    # --- Geography --------------------------------------------------
    {"name": "customer_state", "label": "State", "type": "select",
     "group": "geography", "options": BRAZIL_STATES, "default": "SP"},
    {"name": "region", "label": "Region (Indian State / UT)",
     "type": "select", "group": "geography", "options": REGIONS,
     "default": "Karnataka",
     "help": "Sent to the model exactly as selected — see the honesty "
             "note above REGIONS in this file for why this doesn't "
             "currently influence predictions (model trained on "
             "Brazilian regions)."},
]

FEATURE_GROUPS: list[tuple[str, str]] = [
    ("identity", "Customer"),
    ("activity", "Order Activity"),
    ("payment", "Payment Behavior"),
    ("reviews", "Reviews"),
    ("delivery", "Delivery Performance"),
    ("basket", "Basket & Category"),
    ("geography", "Geography"),
]

# ---------------------------------------------------------------------------
# AI Copilot chat page — suggested prompts
# ---------------------------------------------------------------------------
# Each prompt maps to a real backend call (no free-text LLM endpoint exists
# on the backend today), so labels describe an action rather than a question.

SUGGESTED_PROMPTS: list[dict] = [
    {"label": "Analyze this customer", "action": "analyze_single"},
    {"label": "What's driving their CLV?", "action": "explain_drivers"},
    {"label": "What campaign should I run?", "action": "recommend_campaign"},
    {"label": "Summarize churn risk", "action": "churn_summary"},
]

# ---------------------------------------------------------------------------
# Misc labels
# ---------------------------------------------------------------------------

FIELD_LABELS: dict[str, str] = {f["name"]: f["label"] for f in FEATURE_FIELDS}

NAV_PAGES: list[dict] = [
    {"key": "dashboard", "label": "Dashboard", "icon": "📊"},
    {"key": "prediction", "label": "Customer Prediction", "icon": "🔮"},
    {"key": "batch", "label": "Batch Prediction", "icon": "📁"},
    {"key": "copilot", "label": "AI Copilot", "icon": "🤖"},
    {"key": "explainability", "label": "Explainability", "icon": "🧠"},
    {"key": "segmentation", "label": "Customer Segmentation", "icon": "🧩"},
    {"key": "reports", "label": "Management Reports", "icon": "📑"},
    {"key": "api_status", "label": "API Status", "icon": "⚙️"},
]
