"""
Telco Customer Churn MCP Server
================================
MCP server that exposes the Telco Customer Churn prediction API as a tool.
Requires the FastAPI app to be running locally on port 8000.

Start the FastAPI app first:
    python -m uvicorn src.app.main:app --host 0.0.0.0 --port 8000

Then register this server in Claude Desktop's config:
    {
        "mcpServers": {
            "churn-predictor": {
                "command": "python",
                "args": ["churn_mcp_server.py"],
                "env": {
                    "CHURN_API_URL": "http://localhost:8000"
                }
            }
        }
    }
"""

import os
import json
import httpx
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

# ── Config ────────────────────────────────────────────────────────────────────
API_URL = os.environ.get("CHURN_API_URL", "http://localhost:8000")

mcp = FastMCP("churn_predictor")


# ── Input Model ───────────────────────────────────────────────────────────────
class CustomerData(BaseModel):
    """18 features required by the churn prediction model."""

    # Demographics
    gender: str = Field(..., description="'Male' or 'Female'")
    Partner: str = Field(..., description="'Yes' or 'No' — has a partner")
    Dependents: str = Field(..., description="'Yes' or 'No' — has dependents")

    # Phone services
    PhoneService: str = Field(..., description="'Yes' or 'No'")
    MultipleLines: str = Field(..., description="'Yes', 'No', or 'No phone service'")

    # Internet services
    InternetService: str = Field(..., description="'DSL', 'Fiber optic', or 'No'")
    OnlineSecurity: str = Field(..., description="'Yes', 'No', or 'No internet service'")
    OnlineBackup: str = Field(..., description="'Yes', 'No', or 'No internet service'")
    DeviceProtection: str = Field(..., description="'Yes', 'No', or 'No internet service'")
    TechSupport: str = Field(..., description="'Yes', 'No', or 'No internet service'")
    StreamingTV: str = Field(..., description="'Yes', 'No', or 'No internet service'")
    StreamingMovies: str = Field(..., description="'Yes', 'No', or 'No internet service'")

    # Account information
    Contract: str = Field(..., description="'Month-to-month', 'One year', or 'Two year'")
    PaperlessBilling: str = Field(..., description="'Yes' or 'No'")
    PaymentMethod: str = Field(
        ...,
        description="'Electronic check', 'Mailed check', 'Bank transfer (automatic)', or 'Credit card (automatic)'"
    )

    # Numeric features
    tenure: int = Field(..., description="Number of months the customer has been with the company", ge=0)
    MonthlyCharges: float = Field(..., description="Monthly charges in dollars", ge=0)
    TotalCharges: float = Field(..., description="Total charges to date in dollars", ge=0)


# ── Tools ─────────────────────────────────────────────────────────────────────

@mcp.tool(name="predict_customer_churn", annotations={"readOnlyHint": True, "destructiveHint": False})
async def predict_customer_churn(params: CustomerData) -> str:
    """
    Predict whether a telecom customer is likely to churn based on their
    account and service details. Returns 'Likely to churn' or 'Not likely to churn'.

    Use this when asked about:
    - Customer retention risk
    - Whether a specific customer might leave
    - Churn probability for a given customer profile

    Requires the FastAPI app to be running on localhost:8000.
    """
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{API_URL}/predict",
                json=params.model_dump(),
                timeout=10.0
            )
            response.raise_for_status()
            result = response.json()

            if "error" in result:
                return f"Prediction error: {result['error']}"

            prediction = result.get("prediction", "Unknown")
            # Enrich the response with key risk factors
            risk_factors = []
            if params.Contract == "Month-to-month":
                risk_factors.append("month-to-month contract")
            if params.InternetService == "Fiber optic" and params.OnlineSecurity == "No":
                risk_factors.append("fiber optic with no security add-ons")
            if params.PaymentMethod == "Electronic check":
                risk_factors.append("electronic check payment")
            if params.tenure < 12:
                risk_factors.append(f"low tenure ({params.tenure} months)")

            output = f"Prediction: {prediction}\n"
            if risk_factors and prediction == "Likely to churn":
                output += f"Key risk factors: {', '.join(risk_factors)}"
            elif prediction == "Not likely to churn":
                output += "No major risk factors detected."

            return output

        except httpx.ConnectError:
            return (
                "Error: Cannot connect to the churn prediction API. "
                "Make sure the FastAPI app is running: "
                "python -m uvicorn src.app.main:app --host 0.0.0.0 --port 8000"
            )
        except httpx.TimeoutException:
            return "Error: Request to prediction API timed out."
        except Exception as e:
            return f"Unexpected error: {str(e)}"


@mcp.tool(name="churn_api_health", annotations={"readOnlyHint": True, "destructiveHint": False})
async def churn_api_health() -> str:
    """
    Check if the churn prediction API is running and healthy.
    Call this first before attempting predictions to verify the service is available.
    """
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{API_URL}/", timeout=5.0)
            if response.status_code == 200:
                return f"✅ Churn prediction API is healthy at {API_URL}"
            return f"⚠️ API responded with status {response.status_code}"
        except httpx.ConnectError:
            return f"❌ Cannot reach API at {API_URL}. Is the FastAPI app running?"


# ── Prompts ───────────────────────────────────────────────────────────────────

@mcp.prompt(name="churn_risk_analysis")
def churn_risk_analysis_prompt() -> str:
    """Analyse churn risk for a customer profile and suggest retention actions."""
    return """You are a customer retention analyst with access to an ML-powered churn prediction model.

When asked to analyse a customer:
1. Call churn_api_health first to verify the prediction service is available
2. Call predict_customer_churn with the customer's details
3. Based on the prediction and risk factors, suggest concrete retention actions:
   - For 'Likely to churn': recommend targeted offers (e.g. contract upgrade discount, 
     security add-on bundle, payment method switch incentive)
   - For 'Not likely to churn': suggest upsell opportunities

Always explain WHY the customer is at risk in plain business language, not technical terms.
"""


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run(transport="stdio")