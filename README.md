# CustomerIQ Frontend

A Streamlit frontend for the CustomerIQ FastAPI backend — CLV prediction,
SHAP explainability, AI Copilot insights, segmentation, and executive
reporting.

## Setup

```bash
cd customeriq_frontend
pip install -r requirements.txt
```

## Configure the backend URL

By default the app points at `http://localhost:8000`. Override with:

```bash
export CUSTOMERIQ_API_URL="https://your-api-host"
```

## Run

```bash
streamlit run streamlit_app.py
```

## Pages

| Page | What it does | Backend endpoint(s) |
|---|---|---|
| Dashboard | Portfolio KPIs, segment/churn/tier distribution | (uses cohort data from other pages — no "all customers" endpoint exists) |
| Customer Prediction | Single-customer CLV, tier, segment, churn, drivers | `POST /copilot` |
| Batch Prediction | CSV upload, batch scoring, CSV/JSON download | `POST /upload-csv`, `GET /download-report/...` |
| AI Copilot | Chat-styled insights (real backend calls, no free-text LLM) | `POST /copilot` |
| Explainability | SHAP drivers, waterfall, global importance | `POST /explain`, `POST /explain/batch` |
| Customer Segmentation | Interactive scatter, filters, drill-down | `POST /copilot/batch` |
| Management Reports | Executive report, CSV/Excel (server) + PDF (client) | `POST /management-report` |
| API Status | Live health, model info, endpoint reference | `GET /health`, `GET /model-info`, `GET /` |

## Notes on two deliberate design choices

- **AI Copilot is chat-styled but not a free-text LLM.** The backend has
  no conversational endpoint — every reply in the chat is generated
  from a real `POST /copilot` response, templated into conversational
  text. Suggested prompts map to specific, real analyses.
- **PDF export is generated client-side** (`utils/pdf_export.py`, via
  `fpdf2`) from the same JSON `/management-report` returns, because the
  backend's `export_report()` only supports CSV/Excel/JSON.

## Folder structure

```
customeriq_frontend/
├── streamlit_app.py          # entry point, theme CSS, landing page
├── config.py                 # API connection + theme tokens (THEME)
├── constants.py               # enums, form schema, mirrored from the backend
├── requirements.txt
├── .streamlit/config.toml
├── services/
│   ├── api_client.py          # every HTTP call to the backend
│   └── session_manager.py     # typed st.session_state schema
├── components/                # reusable render functions
│   ├── kpi_card.py
│   ├── charts.py
│   ├── customer_form.py
│   ├── sidebar.py
│   ├── loaders.py
│   └── notifications.py
├── pages/                      # one file per page
│   ├── 1_📊_Dashboard.py
│   ├── 2_🔮_Customer_Prediction.py
│   ├── 3_📁_Batch_Prediction.py
│   ├── 4_🤖_AI_Copilot.py
│   ├── 5_🧠_Explainability.py
│   ├── 6_🧩_Customer_Segmentation.py
│   ├── 7_📑_Management_Reports.py
│   └── 8_⚙️_API_Status.py
└── utils/
    ├── formatting.py
    ├── validators.py
    └── pdf_export.py
```
