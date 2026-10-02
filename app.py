"""
CustomerIQ FastAPI Backend — Production ML Serving Layer.

Exposes the CustomerIQ CLV prediction platform as a RESTful API.
All ML logic lives in the upstream modules; this file is purely a
transport and validation layer.

Startup sequence
----------------
The application loads the model bundle, feature dataset, and SHAP
explainer exactly once in a lifespan context manager.  Every request
handler receives these objects via FastAPI dependency injection —
no model loading happens inside a request handler.

Endpoints
---------
GET  /                          Platform info
GET  /health                    Liveness and readiness probe
GET  /model-info                Model metadata and feature schema
POST /predict                   Single-customer CLV prediction (JSON)
POST /predict/batch             Batch CLV prediction (JSON)
POST /explain                   Single-customer SHAP explanation (JSON)
POST /explain/batch             Batch SHAP explanation (JSON)
POST /copilot                   Single-customer AI insight (JSON)
POST /copilot/batch             Batch AI insights (JSON)
POST /management-report         Executive management report (JSON)
POST /upload-csv                CSV upload → batch prediction
GET  /download-report/{filename} Download a previously generated report

Usage
-----
    uvicorn app:app --reload --host 0.0.0.0 --port 8000
    # Interactive docs: http://localhost:8000/docs
"""

from __future__ import annotations

import io
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

import aiofiles
import numpy as np
import pandas as pd
import uvicorn
from fastapi import (
    Depends,
    FastAPI,
    File,
    HTTPException,
    Request,
    Response,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, field_validator, model_validator

# Project root on sys.path so src.* imports resolve when running from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data.constants import MAX_ROWS, MODELS_DIR, REPORTS_DIR
from src.models.ai_copilot import (
    AIInsight,
    ExportFormat,
    analyze_batch,
    analyze_customer,
    export_report,
    generate_management_report,
    recommend_campaigns,
    segment_customers,
)
from src.models.explainability import (
    ExplainerBundle,
    SingleExplanation,
    build_explainer,
    explain_batch_predictions,
    explain_single_prediction,
    global_feature_importance,
)
from src.models.prediction import (
    InputValidationError,
    ModelBundle,
    ModelNotFoundError,
    PredictionError,
    load_saved_model,
    prediction_pipeline,
    preprocess_prediction_input,
    validate_prediction_input,
)
from src.models.model_training import load_feature_dataset, prepare_features_target
from src.utils.logging_config import get_logger, setup_logging

setup_logging("INFO", json_output=True)
logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Application state — loaded once at startup
# ---------------------------------------------------------------------------


class _AppState:
    """Singleton holder for startup-loaded ML objects."""

    bundle: ModelBundle
    explainer_bundle: ExplainerBundle
    feature_reference: pd.DataFrame   # full feature dataset for segment context


_state = _AppState()


# ---------------------------------------------------------------------------
# Lifespan (replaces deprecated @app.on_event)
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all ML objects before the first request; release on shutdown."""
    logger.info("CustomerIQ API starting up — loading ML artefacts.")
    t0 = time.perf_counter()

    _state.bundle = load_saved_model(prefer_tuned=True)

    # Load the feature dataset once so ai_copilot can access raw feature rows.
    # (Loaded before build_explainer so a background sample can be passed to
    # it — see build_explainer()'s docstring re: SHAP + LightGBM categorical
    # splits leaving expected_value unset without one.)
    df_raw = load_feature_dataset()
    _state.feature_reference = df_raw

    _background_sample = prepare_features_target(df_raw)[0][_state.bundle.feature_names].sample(
        n=min(200, len(df_raw)), random_state=42
    )
    _state.explainer_bundle = build_explainer(_state.bundle, background_data=_background_sample)

    elapsed = round(time.perf_counter() - t0, 2)
    logger.info(
        "Startup complete | model='%s' | features=%d | time=%.2fs",
        _state.bundle.model_version,
        len(_state.bundle.feature_names),
        elapsed,
    )
    yield

    logger.info("CustomerIQ API shutting down.")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

_DESCRIPTION = """
**CustomerIQ** — Production-grade Customer Lifetime Value prediction API.

Powered by LightGBM + SHAP + Optuna.

### Capabilities
- **Predict** 90-day Customer Lifetime Value (CLV) for individual or batch customers.
- **Explain** predictions using SHAP (SHapley Additive exPlanations).
- **AI Copilot** — automated segmentation, churn risk, campaign recommendations, and executive summaries.
- **Upload CSV** — batch-score customers from a CSV file without writing code.
- **Download Reports** — export insights as CSV, Excel, or JSON.

### Notes
- All monetary values are in Brazilian Real (BRL).
- Predictions are on the log₁p-transformed scale internally; all API responses return the raw BRL value.
- The model was trained on the [Olist Brazilian E-Commerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) dataset.
"""

app = FastAPI(
    title="CustomerIQ CLV Platform",
    version="1.0.0",
    description=_DESCRIPTION,
    contact={
        "name": "CustomerIQ Engineering",
        "email": "ml-platform@customeriq.ai",
    },
    license_info={"name": "MIT"},
    openapi_tags=[
        {"name": "Core", "description": "Health, info, and root endpoints."},
        {"name": "Predict", "description": "CLV prediction endpoints."},
        {"name": "Explain", "description": "SHAP explanation endpoints."},
        {"name": "Copilot", "description": "AI business insight endpoints."},
        {"name": "Reports", "description": "File upload, download, and report generation."},
    ],
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

app.add_middleware(GZipMiddleware, minimum_size=1000)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def _request_logging_middleware(request: Request, call_next):
    """Log every request with method, path, status, and latency."""
    request_id = str(uuid.uuid4())[:8]
    request.state.request_id = request_id
    t0 = time.perf_counter()

    logger.info(
        "Request started | id=%s | %s %s",
        request_id,
        request.method,
        request.url.path,
    )

    response: Response = await call_next(request)

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-Ms"] = str(elapsed_ms)

    logger.info(
        "Request complete | id=%s | status=%d | time=%.1fms",
        request_id,
        response.status_code,
        elapsed_ms,
    )
    return response


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------


@app.exception_handler(ModelNotFoundError)
async def _model_not_found_handler(request: Request, exc: ModelNotFoundError):
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"error": "model_not_found", "detail": str(exc)},
    )


@app.exception_handler(InputValidationError)
async def _input_validation_handler(request: Request, exc: InputValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"error": "input_validation_error", "detail": str(exc)},
    )


@app.exception_handler(PredictionError)
async def _prediction_error_handler(request: Request, exc: PredictionError):
    logger.error("PredictionError: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "prediction_error", "detail": str(exc)},
    )


@app.exception_handler(Exception)
async def _generic_error_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception: %s — %s", type(exc).__name__, exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "internal_server_error", "detail": str(exc)},
    )


# ---------------------------------------------------------------------------
# Dependency injection
# ---------------------------------------------------------------------------


def get_bundle() -> ModelBundle:
    """Inject the loaded ModelBundle into route handlers."""
    return _state.bundle


def get_explainer() -> ExplainerBundle:
    """Inject the loaded ExplainerBundle into route handlers."""
    return _state.explainer_bundle


def get_feature_reference() -> pd.DataFrame:
    """Inject the raw feature reference DataFrame."""
    return _state.feature_reference


BundleDep = Annotated[ModelBundle, Depends(get_bundle)]
ExplainerDep = Annotated[ExplainerBundle, Depends(get_explainer)]
FeatureRefDep = Annotated[pd.DataFrame, Depends(get_feature_reference)]


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------


class CustomerFeatures(BaseModel):
    """Feature vector for a single customer.

    All features mirror the output of ``feature_engineering.py``.
    Categorical fields accept the string values observed during training.
    Missing numeric values should be omitted (they default to ``None``
    and are handled natively by LightGBM).
    """

    model_config = {"populate_by_name": True}

    customer_unique_id: str | None = Field(None, description="Customer identifier (not used for prediction).")

    # Numeric features
    recency_days: float | None = Field(None, ge=0, description="Days since last delivered order.")
    frequency: int | None = Field(None, ge=1, description="Number of delivered orders.")
    total_orders: int | None = Field(None, ge=1)
    delivered_orders: int | None = Field(None, ge=0)
    cancelled_orders: int | None = Field(None, ge=0)
    order_success_rate: float | None = Field(None, ge=0.0, le=1.0)
    avg_installments: float | None = Field(None, ge=1.0)
    pct_credit_card_payments: float | None = Field(None, ge=0.0, le=1.0)
    avg_review_score: float | None = Field(None, ge=1.0, le=5.0)
    min_review_score: float | None = Field(None, ge=1.0, le=5.0)
    max_review_score: float | None = Field(None, ge=1.0, le=5.0)
    review_count: float | None = Field(None, ge=0)
    pct_low_reviews: float | None = Field(None, ge=0.0, le=1.0)
    pct_five_star_reviews: float | None = Field(None, ge=0.0, le=1.0)
    avg_delivery_days: float | None = Field(None, ge=0)
    avg_shipping_days: float | None = Field(None, ge=0)
    late_delivery_pct: float | None = Field(None, ge=0.0, le=1.0)
    early_delivery_pct: float | None = Field(None, ge=0.0, le=1.0)
    n_delivered_with_dates: int | None = Field(None, ge=0)
    customer_lifetime_days: int | None = Field(None, ge=0)
    tenure_days: int | None = Field(None, ge=0)
    avg_items_per_order: float | None = Field(None, ge=0)
    avg_freight_value: float | None = Field(None, ge=0)
    total_items_purchased: int | None = Field(None, ge=0)
    unique_categories: int | None = Field(None, ge=1)
    category_hhi: float | None = Field(None, ge=0.0, le=1.0)

    # Categorical features
    most_common_payment_type: str | None = Field(
        None,
        description="One of: credit_card, boleto, voucher, debit_card, not_defined",
    )
    customer_state: str | None = Field(None, description="Brazilian state code (e.g. SP, RJ).")
    region: str | None = Field(
        None,
        description="Macro-region: Southeast, South, Northeast, North, Midwest.",
    )
    top_category: str | None = Field(None, description="English product category name.")

    @field_validator("most_common_payment_type")
    @classmethod
    def _validate_payment_type(cls, v: str | None) -> str | None:
        valid = {"credit_card", "boleto", "voucher", "debit_card", "not_defined"}
        if v is not None and v not in valid:
            raise ValueError(f"most_common_payment_type must be one of {valid}.")
        return v

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to a single-row DataFrame suitable for prediction."""
        return pd.DataFrame([self.model_dump()])


class BatchCustomerFeatures(BaseModel):
    """Request body for batch prediction endpoints."""

    customers: list[CustomerFeatures] = Field(
        ...,
        min_length=1,
        max_length=MAX_ROWS,
        description=f"List of customer feature objects (1-{MAX_ROWS:,} rows).",
    )

    @model_validator(mode="after")
    def _check_not_empty(self) -> "BatchCustomerFeatures":
        if not self.customers:
            raise ValueError("customers list must not be empty.")
        return self

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([c.model_dump() for c in self.customers])


class PredictionResponse(BaseModel):
    """Single-customer prediction response."""

    customer_id: str | None
    predicted_clv_log_scale: float
    predicted_clv_brl: float
    clv_tier: str
    model_version: str
    prediction_timestamp_utc: str


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""

    n_customers: int
    model_version: str
    predictions: list[dict[str, Any]]
    warnings: list[str] = []


class DriverItem(BaseModel):
    """A single SHAP driver (positive or negative)."""

    feature: str
    shap_value: float
    feature_value: Any


class ExplanationResponse(BaseModel):
    """Single-customer SHAP explanation response."""

    customer_id: str | None
    base_value: float
    prediction_log_scale: float
    prediction_brl: float
    top_positive_drivers: list[DriverItem]
    top_negative_drivers: list[DriverItem]
    model_version: str
    computation_time_sec: float


class BatchExplanationResponse(BaseModel):
    """Batch SHAP explanation response — returns global importance."""

    n_customers: int
    base_value: float
    model_version: str
    global_feature_importance: list[dict[str, Any]]
    computation_time_sec: float


class RevenueOpportunityResponse(BaseModel):
    predicted_clv_brl: float
    upsell_opportunity_brl: float
    retention_value_brl: float
    cross_sell_lift_brl: float
    priority_score: float
    total_opportunity_brl: float


class CopilotResponse(BaseModel):
    """Single-customer AI copilot response."""

    customer_id: str
    predicted_clv: float
    clv_tier: str
    segment: str
    churn_risk: str
    churn_risk_score: float
    business_summary: str
    nl_explanation: str
    top_positive_drivers: list[DriverItem]
    top_negative_drivers: list[DriverItem]
    recommended_actions: list[str]
    revenue_opportunity: RevenueOpportunityResponse
    confidence_score: float
    model_version: str
    generated_at_utc: str


class ManagementReportRequest(BaseModel):
    """Request body for the management report endpoint."""

    customers: list[CustomerFeatures] = Field(
        ..., min_length=1, max_length=MAX_ROWS
    )
    include_global_importance: bool = Field(
        True, description="Include SHAP global importance in the report."
    )
    export_format: str | None = Field(
        None,
        description="If set, export the report file. One of: csv, excel, json.",
    )

    @field_validator("export_format")
    @classmethod
    def _validate_fmt(cls, v: str | None) -> str | None:
        if v is not None and v not in {"csv", "excel", "json"}:
            raise ValueError("export_format must be one of: csv, excel, json.")
        return v


class HealthResponse(BaseModel):
    status: str
    model_version: str
    n_features: int
    uptime_sec: float


class ModelInfoResponse(BaseModel):
    model_version: str
    target_column: str
    n_features: int
    feature_names: list[str]
    categorical_features: list[str]
    known_categories: dict[str, list[str]]
    hyperparameters: dict[str, Any]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_STARTUP_TIME = time.time()


def _insight_to_response(insight: AIInsight) -> CopilotResponse:
    """Convert an AIInsight dataclass to a CopilotResponse Pydantic model."""
    ro = insight.revenue_opportunity
    return CopilotResponse(
        customer_id=insight.customer_id,
        predicted_clv=insight.predicted_clv,
        clv_tier=insight.clv_tier,
        segment=insight.segment.value,
        churn_risk=insight.churn_risk.value,
        churn_risk_score=insight.churn_risk_score,
        business_summary=insight.business_summary,
        nl_explanation=insight.nl_explanation,
        top_positive_drivers=[DriverItem(**d) for d in insight.top_positive_drivers],
        top_negative_drivers=[DriverItem(**d) for d in insight.top_negative_drivers],
        recommended_actions=[c.value for c in insight.recommended_actions],
        revenue_opportunity=RevenueOpportunityResponse(
            predicted_clv_brl=ro.predicted_clv_brl,
            upsell_opportunity_brl=ro.upsell_opportunity_brl,
            retention_value_brl=ro.retention_value_brl,
            cross_sell_lift_brl=ro.cross_sell_lift_brl,
            priority_score=ro.priority_score,
            total_opportunity_brl=ro.total_opportunity_brl,
        ),
        confidence_score=insight.confidence_score,
        model_version=insight.model_version,
        generated_at_utc=insight.generated_at_utc,
    )


def _df_to_input(df: pd.DataFrame, bundle: ModelBundle) -> pd.DataFrame:
    """Validate and return the input DataFrame; raises HTTPException on failure."""
    try:
        validate_prediction_input(df, bundle)
    except InputValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )
    return df


def _run_lightweight_copilot(
    input_df: pd.DataFrame,
    bundle: ModelBundle,
    feature_ref: pd.DataFrame,
) -> list[AIInsight]:
    """Run prediction + copilot WITHOUT per-row SHAP explanation.

    Used by the primary batch-scale workflow (Batch Prediction page's
    default flow, and /upload-csv) where the goal is fast prediction +
    segmentation + churn risk for up to MAX_ROWS customers, not a
    per-customer driver breakdown. Segment/churn-risk/campaigns/revenue-
    opportunity are all computed from the raw feature row and predicted
    CLV alone — none of them depend on SHAP (see
    ai_copilot.analyze_customer's internal call graph). Only
    `nl_explanation` (falls back to a driver-free template) and
    `confidence_score` (falls back to a non-SHAP heuristic) are mildly
    degraded versus the full path; `top_positive_drivers` /
    `top_negative_drivers` come back as empty lists, which is honest —
    no SHAP was computed for this call.

    For a genuine per-customer driver breakdown, call `/copilot` (single)
    or `/explain` — both remain full-SHAP and fast at n=1.
    """
    pred_result = prediction_pipeline(input_df)
    raw_pred = pred_result["predictions"]
    pred_df = raw_pred if isinstance(raw_pred, pd.DataFrame) else pd.DataFrame([raw_pred.as_dict()])

    id_col = "customer_unique_id"
    if id_col in input_df.columns:
        feat_rows = feature_ref.set_index(id_col).reindex(
            input_df[id_col].values
        ).reset_index()
    else:
        feat_rows = input_df.reset_index(drop=True)

    return analyze_batch(
        prediction_df=pred_df,
        feature_df=feat_rows,
        batch_shap_values=None,
        feature_names=None,
    )


def _run_full_copilot(
    input_df: pd.DataFrame,
    bundle: ModelBundle,
    eb: ExplainerBundle,
    feature_ref: pd.DataFrame,
) -> list[AIInsight]:
    """Run prediction + SHAP + copilot for a batch of customers."""
    pred_result = prediction_pipeline(input_df)

    # prediction_pipeline returns SinglePredictionResult for n=1,
    # and a DataFrame for n>1.  Normalise to DataFrame for analyze_batch.
    raw_pred = pred_result["predictions"]
    if isinstance(raw_pred, pd.DataFrame):
        pred_df = raw_pred
    else:
        # SinglePredictionResult — convert to single-row DataFrame.
        pred_df = pd.DataFrame([raw_pred.as_dict()])

    validate_prediction_input(input_df, bundle)
    preprocessed = preprocess_prediction_input(input_df, bundle)
    batch_expl = explain_batch_predictions(preprocessed, eb)

    # Best-effort feature row alignment from the reference dataset.
    id_col = "customer_unique_id"
    if id_col in input_df.columns:
        feat_rows = feature_ref.set_index(id_col).reindex(
            input_df[id_col].values
        ).reset_index()
    else:
        feat_rows = input_df.reset_index(drop=True)

    return analyze_batch(
        prediction_df=pred_df,
        feature_df=feat_rows,
        batch_shap_values=batch_expl.shap_values,
        feature_names=batch_expl.feature_names,
    )


# ---------------------------------------------------------------------------
# Routes — Core
# ---------------------------------------------------------------------------


@app.get("/", tags=["Core"], summary="Platform root")
def root():
    """Return a brief platform description and links to documentation."""
    return {
        "platform": "CustomerIQ CLV Prediction API",
        "version": app.version,
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/health",
        "model_info": "/model-info",
    }


@app.get(
    "/health",
    tags=["Core"],
    summary="Liveness and readiness probe",
    response_model=HealthResponse,
)
def health(bundle: BundleDep):
    """Return service health status.

    Returns HTTP 200 when the model is loaded and ready.
    Returns HTTP 503 if the model bundle failed to load at startup.
    """
    return HealthResponse(
        status="healthy",
        model_version=bundle.model_version,
        n_features=len(bundle.feature_names),
        uptime_sec=round(time.time() - _STARTUP_TIME, 1),
    )


@app.get(
    "/model-info",
    tags=["Core"],
    summary="Model metadata and feature schema",
    response_model=ModelInfoResponse,
)
def model_info(bundle: BundleDep):
    """Return the loaded model's metadata, feature list, and hyperparameters.

    Use this endpoint to verify which model version is active and to
    discover the full feature schema expected by ``/predict``.
    """
    known_cats = {
        col: list(idx.astype(str))
        for col, idx in bundle.known_categories.items()
    }
    # Strip non-serialisable params (n_jobs, verbose, etc.).
    safe_params = {
        k: v for k, v in bundle.params.items()
        if isinstance(v, (str, int, float, bool, type(None)))
    }
    return ModelInfoResponse(
        model_version=bundle.model_version,
        target_column=bundle.target_column,
        n_features=len(bundle.feature_names),
        feature_names=bundle.feature_names,
        categorical_features=bundle.categorical_features,
        known_categories=known_cats,
        hyperparameters=safe_params,
    )


# ---------------------------------------------------------------------------
# Routes — Predict
# ---------------------------------------------------------------------------


@app.post(
    "/predict",
    tags=["Predict"],
    summary="Predict CLV for a single customer",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
)
def predict_single(
    body: CustomerFeatures,
    bundle: BundleDep,
):
    """Predict the 90-day Customer Lifetime Value for one customer.

    Submit a JSON object with customer feature values.  All features are
    optional — missing values are handled natively by the model.

    Returns the predicted CLV in BRL, the CLV tier, and the model version.
    """
    input_df = body.to_dataframe()
    _df_to_input(input_df, bundle)

    result = prediction_pipeline(input_df)
    pred = result["predictions"]   # SinglePredictionResult for n=1

    return PredictionResponse(
        customer_id=pred.customer_id,
        predicted_clv_log_scale=pred.predicted_clv_log_scale,
        predicted_clv_brl=pred.predicted_clv_brl,
        clv_tier=pred.clv_tier,
        model_version=pred.model_version,
        prediction_timestamp_utc=pred.prediction_timestamp_utc,
    )
@app.get(
    "/customers/{customer_id}",
    tags=["Customers"],
    summary="Get customer feature data by ID",
)
async def get_customer(
    customer_id: str,
    feature_ref: FeatureRefDep,
):
    """Return the stored feature values for one customer."""
    id_col = "customer_unique_id"

    if id_col not in feature_ref.columns:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Customer ID column is missing from the feature dataset.",
        )

    matches = feature_ref[
        feature_ref[id_col].astype(str).str.strip() == customer_id.strip()
    ]

    if matches.empty:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Customer '{customer_id}' not found.",
        )

    row = matches.iloc[0]

    return {
        key: None if pd.isna(value) else value
        for key, value in row.to_dict().items()
    }

@app.post(
    "/predict/batch",
    tags=["Predict"],
    summary="Predict CLV for multiple customers",
    response_model=BatchPredictionResponse,
    status_code=status.HTTP_200_OK,
)
def predict_batch(
    body: BatchCustomerFeatures,
    bundle: BundleDep,
):
    """Predict 90-day CLV for a batch of 1-10,000 customers.

    Submit a JSON array under the ``customers`` key.  Returns one
    prediction row per input customer, preserving input order.
    """
    input_df = body.to_dataframe()
    _df_to_input(input_df, bundle)

    result = prediction_pipeline(input_df)
    pred_df: pd.DataFrame = result["predictions"]

    return BatchPredictionResponse(
        n_customers=result["n_customers"],
        model_version=result["model_version"],
        predictions=pred_df.to_dict(orient="records"),
        warnings=result.get("warnings", []),
    )


# ---------------------------------------------------------------------------
# Routes — Explain
# ---------------------------------------------------------------------------


@app.post(
    "/explain",
    tags=["Explain"],
    summary="SHAP explanation for a single customer",
    response_model=ExplanationResponse,
    status_code=status.HTTP_200_OK,
)
def explain_single(
    body: CustomerFeatures,
    bundle: BundleDep,
    eb: ExplainerDep,
):
    """Return SHAP feature contributions for one customer prediction.

    The explanation shows which features pushed the predicted CLV above
    or below the model baseline, ranked by absolute SHAP value.

    All SHAP values are on the log₁p scale (the model's training scale).
    ``prediction_brl`` is the expm1-transformed CLV in BRL.
    """
    input_df = body.to_dataframe()
    _df_to_input(input_df, bundle)

    validate_prediction_input(input_df, bundle)
    preprocessed = preprocess_prediction_input(input_df, bundle)
    expl: SingleExplanation = explain_single_prediction(
        preprocessed_row=preprocessed,
        explainer_bundle=eb,
        customer_id=body.customer_unique_id,
    )

    return ExplanationResponse(
        customer_id=expl.customer_id,
        base_value=expl.base_value,
        prediction_log_scale=expl.prediction_log_scale,
        prediction_brl=expl.prediction_brl,
        top_positive_drivers=[DriverItem(**d) for d in expl.top_positive_drivers],
        top_negative_drivers=[DriverItem(**d) for d in expl.top_negative_drivers],
        model_version=expl.model_version,
        computation_time_sec=expl.computation_time_sec,
    )


@app.post(
    "/explain/batch",
    tags=["Explain"],
    summary="SHAP global importance for a batch",
    response_model=BatchExplanationResponse,
    status_code=status.HTTP_200_OK,
)
def explain_batch(
    body: BatchCustomerFeatures,
    bundle: BundleDep,
    eb: ExplainerDep,
):
    """Compute SHAP values for a batch and return global feature importance.

    Returns the mean absolute SHAP value per feature across all customers
    in the batch, ranked descending.  This gives a population-level view
    of what drives CLV for a specific cohort.

    For per-customer SHAP values, use ``/explain`` in a loop or the
    ``/copilot/batch`` endpoint which includes per-customer driver lists.
    """
    input_df = body.to_dataframe()
    _df_to_input(input_df, bundle)

    validate_prediction_input(input_df, bundle)
    preprocessed = preprocess_prediction_input(input_df, bundle)
    batch_expl = explain_batch_predictions(preprocessed, eb)
    gi = global_feature_importance(batch_expl)

    return BatchExplanationResponse(
        n_customers=batch_expl.n_customers,
        base_value=batch_expl.base_value,
        model_version=batch_expl.model_version,
        global_feature_importance=gi.importance_df[
            ["feature", "mean_abs_shap", "mean_abs_shap_norm", "rank"]
        ].to_dict(orient="records"),
        computation_time_sec=batch_expl.computation_time_sec,
    )


# ---------------------------------------------------------------------------
# Routes — AI Copilot
# ---------------------------------------------------------------------------


@app.post(
    "/copilot",
    tags=["Copilot"],
    summary="AI business insight for a single customer",
    response_model=CopilotResponse,
    status_code=status.HTTP_200_OK,
)
def copilot_single(
    body: CustomerFeatures,
    bundle: BundleDep,
    eb: ExplainerDep,
    feature_ref: FeatureRefDep,
):
    """Generate a complete AI business insight for one customer.

    Combines CLV prediction, SHAP explanation, customer segmentation,
    churn risk estimation, campaign recommendations, revenue opportunity
    estimate, and a natural-language executive summary.
    """
    input_df = body.to_dataframe()
    _df_to_input(input_df, bundle)

    insights = _run_full_copilot(input_df, bundle, eb, feature_ref)
    return _insight_to_response(insights[0])


@app.post(
    "/copilot/batch",
    tags=["Copilot"],
    summary="AI business insights for multiple customers",
    response_model=list[CopilotResponse],
    status_code=status.HTTP_200_OK,
)
def copilot_batch(
    body: BatchCustomerFeatures,
    bundle: BundleDep,
    eb: ExplainerDep,
    feature_ref: FeatureRefDep,
):
    """Generate AI business insights for a batch of 1-10,000 customers.

    Full-fidelity path: computes real per-customer SHAP drivers via
    _run_full_copilot, so every CopilotResponse has populated
    top_positive_drivers/top_negative_drivers. This is genuinely
    expensive at scale (SHAP over the whole batch) — for the primary
    "score many customers fast" workflow (Batch Prediction page),
    prefer POST /copilot/batch/lightweight instead, which skips SHAP
    and returns in a fraction of the time.
    """
    input_df = body.to_dataframe()
    _df_to_input(input_df, bundle)

    insights = _run_full_copilot(input_df, bundle, eb, feature_ref)
    return [_insight_to_response(ins) for ins in insights]


@app.post(
    "/copilot/batch/lightweight",
    tags=["Copilot"],
    summary="Fast batch scoring: prediction + segmentation + churn, no per-row SHAP",
    response_model=list[CopilotResponse],
    status_code=status.HTTP_200_OK,
)
def copilot_batch_lightweight(
    body: BatchCustomerFeatures,
    bundle: BundleDep,
    feature_ref: FeatureRefDep,
):
    """Score up to 10,000 customers without computing per-row SHAP.

    This is the endpoint the Batch Prediction page uses by default: CLV,
    CLV tier, segment, and churn risk are all populated normally (none
    of them depend on SHAP), but `top_positive_drivers` /
    `top_negative_drivers` come back as empty lists on every row — no
    SHAP was computed, so there's nothing honest to put there. For a
    specific customer's driver breakdown, call `/copilot` (single) or
    `/explain`, both of which remain fast at n=1 with full SHAP.

    Roughly an order of magnitude faster than /copilot/batch at 10,000
    rows, since it skips the batch SHAP pass entirely rather than just
    running it more efficiently.
    """
    input_df = body.to_dataframe()
    _df_to_input(input_df, bundle)

    insights = _run_lightweight_copilot(input_df, bundle, feature_ref)
    return [_insight_to_response(ins) for ins in insights]


@app.post(
    "/management-report",
    tags=["Copilot"],
    summary="Executive management report",
    status_code=status.HTTP_200_OK,
)
def management_report(
    body: ManagementReportRequest,
    bundle: BundleDep,
    eb: ExplainerDep,
    feature_ref: FeatureRefDep,
):
    """Generate an executive management report for a customer cohort.

    Returns a structured summary including:
    - CLV distribution statistics
    - Segment and churn risk distribution
    - Campaign prioritisation breakdown
    - Top 10 revenue opportunity customers
    - Model health metrics
    - Top global SHAP features (optional)

    Optionally exports the report to a file (``export_format`` field).
    The download URL for the file is returned in the response.
    """
    batch_df = pd.DataFrame([c.model_dump() for c in body.customers])
    _df_to_input(batch_df, bundle)

    insights = _run_full_copilot(batch_df, bundle, eb, feature_ref)

    gi_df: pd.DataFrame | None = None
    if body.include_global_importance:
        validate_prediction_input(batch_df, bundle)
        preprocessed = preprocess_prediction_input(batch_df, bundle)
        batch_expl = explain_batch_predictions(preprocessed, eb)
        gi = global_feature_importance(batch_expl)
        gi_df = gi.importance_df

    report = generate_management_report(insights, global_importance_df=gi_df)

    download_url: str | None = None
    if body.export_format:
        fmt_map = {"csv": ExportFormat.CSV, "excel": ExportFormat.EXCEL, "json": ExportFormat.JSON}
        fmt = fmt_map[body.export_format]
        export_path = export_report(insights, fmt=fmt, filename_prefix="mgmt_report")
        download_url = f"/download-report/{export_path.parent.name}/{export_path.name}"

    return {**report, "download_url": download_url}


# ---------------------------------------------------------------------------
# Routes — Reports
# ---------------------------------------------------------------------------


@app.post(
    "/upload-csv",
    tags=["Reports"],
    summary="Upload a CSV file for batch prediction",
    status_code=status.HTTP_200_OK,
)
async def upload_csv(
    bundle: BundleDep,
    feature_ref: FeatureRefDep,
    file: UploadFile = File(..., description="CSV file with customer feature columns."),
):
    """Upload a CSV file to run batch CLV prediction and AI analysis.

    The CSV must include at least a subset of the model's 30 feature
    columns.  Use ``GET /model-info`` to see the full feature schema.

    Returns:
    - CLV predictions for every row
    - Customer segmentation
    - Campaign recommendations
    - Download URLs for CSV and JSON exports
    """
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only .csv files are supported. Ensure the file has a .csv extension.",
        )

    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Uploaded file is empty.",
        )

    try:
        input_df = pd.read_csv(io.BytesIO(contents))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Could not parse CSV: {exc}",
        )

    if input_df.empty:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="CSV file contains no data rows.",
        )

    if len(input_df) > MAX_ROWS:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"CSV contains {len(input_df):,} rows; maximum is {MAX_ROWS:,} per upload.",
        )

    _df_to_input(input_df, bundle)
    logger.info("CSV upload | file='%s' | rows=%d", file.filename, len(input_df))

    # Heavy synchronous work (prediction + segmentation, no per-row SHAP —
    # see _run_lightweight_copilot's docstring) is offloaded to FastAPI's
    # threadpool via run_in_threadpool rather than run inline: this is an
    # `async def` route (it needs `await file.read()` above), so calling
    # CPU-bound code directly here would still block the event loop for
    # the full duration, exactly like the other routes did before they
    # were converted to plain `def`.
    def _process() -> list[AIInsight]:
        return _run_lightweight_copilot(input_df, bundle, feature_ref)

    insights = await run_in_threadpool(_process)

    seg_df = segment_customers(insights)
    camp_df = recommend_campaigns(insights, top_n_per_campaign=20)

    csv_path = export_report(insights, fmt=ExportFormat.CSV, filename_prefix="csv_upload")
    json_path = export_report(insights, fmt=ExportFormat.JSON, filename_prefix="csv_upload")

    return {
        "status": "success",
        "source_file": file.filename,
        "n_customers_scored": len(insights),
        "segment_distribution": seg_df["segment"].value_counts().to_dict(),
        "churn_risk_distribution": seg_df["churn_risk"].value_counts().to_dict(),
        "total_predicted_revenue_brl": round(
            sum(i.predicted_clv for i in insights), 2
        ),
        "total_opportunity_brl": round(
            sum(i.revenue_opportunity.total_opportunity_brl for i in insights), 2
        ),
        "top_5_priority_customers": seg_df.head(5)[
            ["customer_id", "segment", "predicted_clv_brl", "priority_score"]
        ].to_dict(orient="records"),
        "downloads": {
            "csv": f"/download-report/copilot/{csv_path.name}",
            "json": f"/download-report/copilot/{json_path.name}",
        },
    }


@app.get(
    "/download-report/{subdir}/{filename}",
    tags=["Reports"],
    summary="Download a generated report file",
)
def download_report(subdir: str, filename: str):
    """Download a previously generated report file by name.

    ``subdir`` is the sub-directory under ``reports/`` where the file
    lives (e.g. ``copilot``, ``explainability``).

    The download URL is returned by ``/upload-csv`` and
    ``/management-report``.  Files are retained on disk until the
    container restarts.
    """
    safe_subdir = Path(subdir).name  # prevent path traversal
    safe_filename = Path(filename).name
    file_path = REPORTS_DIR / safe_subdir / safe_filename

    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Report file '{safe_filename}' not found. "
                   "Reports are ephemeral and cleared on server restart.",
        )

    # Infer media type from extension.
    ext = file_path.suffix.lower()
    media_type_map = {
        ".csv": "text/csv",
        ".json": "application/json",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".png": "image/png",
        ".html": "text/html",
    }
    media_type = media_type_map.get(ext, "application/octet-stream")

    return FileResponse(
        path=file_path,
        filename=safe_filename,
        media_type=media_type,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("RELOAD", "true").lower() == "true",
        log_level="info",
        access_log=False,   # handled by our middleware
    )
