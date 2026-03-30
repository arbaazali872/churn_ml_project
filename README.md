# Telco Customer Churn Prediction

End-to-end MLOps pipeline for predicting customer churn in the telecom industry, with a FastAPI serving layer and an MCP server for natural language access via AI agents.

## What it does

Train an XGBoost model on telecom customer data, serve predictions via a REST API, and interact with it through any MCP-compatible AI client — ask in plain English whether a customer is likely to churn.

> *"Predict churn for a customer on a month-to-month contract, fiber optic internet, tenure 1 month"*

The agent calls the model and returns the prediction with key risk factors explained in business terms.

## Stack

- **Model**: XGBoost with MLflow tracking and artifact logging
- **Serving**: FastAPI (`/predict`) + MLflow pyfunc model loading
- **Agent interface**: MCP server (`churn_mcp_server.py`) wrapping the REST API
- **Validation**: Great Expectations for data quality checks
- **Infra**: Docker, GitHub Actions CI/CD → Docker Hub → AWS Fargate

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model
python scripts/run_pipeline.py --input data/raw/Telco-Customer-Churn.csv --target Churn

# 3. Start the API (Python 3.12+: use app_api_only to avoid Gradio/distutils issues)
python -m uvicorn src.app.app_api_only:app --host 0.0.0.0 --port 8000

# 4. Test the endpoint
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"gender":"Female","tenure":1,"Contract":"Month-to-month",...}'
```

## MCP Server

`churn_mcp_server.py` exposes the prediction API as an MCP tool, letting any MCP-compatible AI client predict churn in natural language.

**Install as a Claude Desktop extension:**

```bash
mcpb pack
# Then: Settings → Extensions → Install Extension → select churn-predictor-mcp.mcpb
```

Start the FastAPI before using the extension. The MCP server includes a `churn_api_health` tool to verify connectivity, and a `churn_risk_analysis` prompt that guides the agent to explain predictions and suggest retention actions.

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| POST | `/predict` | Predict churn for a customer (18 features) |

Accepts a `CustomerData` payload with demographics, service subscriptions, contract type, and billing info. Returns `"Likely to churn"` or `"Not likely to churn"`.

## ML Pipeline

```
Data Loading → Validation (Great Expectations) → Preprocessing
→ Feature Engineering → XGBoost Training → MLflow Logging
```

Training and serving use identical feature transformations to prevent train/serve skew — binary encoding via a fixed `BINARY_MAP`, one-hot encoding with `drop_first=True`, and feature alignment enforced by `feature_columns.txt` from training artifacts.

## Docker

```bash
docker build -t telco-churn-app .
docker run -p 8000:8000 telco-churn-app
```

CI/CD via GitHub Actions builds and pushes to Docker Hub on every push to main. Deployment is manual via AWS ECS (Fargate + ALB).