"""
Telco Customer Churn Prediction API - Minimal serving app (no Gradio)
=====================================================================
Use this instead of main.py if you have Python 3.12+ where distutils
was removed and Gradio breaks.

Run with:
    python -m uvicorn src.app.app_api_only:app --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI
from pydantic import BaseModel
from src.serving.inference import predict

app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description="ML API for predicting customer churn in telecom industry",
    version="1.0.0"
)


@app.get("/")
def root():
    return {"status": "ok"}


class CustomerData(BaseModel):
    # Demographics
    gender: str
    Partner: str
    Dependents: str
    # Phone services
    PhoneService: str
    MultipleLines: str
    # Internet services
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    # Account information
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    # Numeric features
    tenure: int
    MonthlyCharges: float
    TotalCharges: float


@app.post("/predict")
def get_prediction(data: CustomerData):
    try:
        result = predict(data.dict())
        return {"prediction": result}
    except Exception as e:
        return {"error": str(e)}