# API Examples

Complete examples for using the Loan Default Prediction API.

## Table of Contents
- [Authentication](#authentication)
- [Making Predictions](#making-predictions)
- [Monitoring Drift](#monitoring-drift)
- [Health Checks](#health-checks)
- [Batch Processing](#batch-processing)

## Base URL

```
http://localhost:8000
```

## Authentication

Currently, the API does not require authentication. For production deployment, implement API keys or OAuth.

## Making Predictions

### Single Prediction

**Endpoint:** `POST /predict`

**Request:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "loan_limit": "cf",
    "loan_type": "type1",
    "loan_purpose": "p4",
    "loan_amount": 250000,
    "property_value": 450000,
    "construction_type": "sb",
    "occupancy_type": "pr",
    "Gender": "Male",
    "age": "35-44",
    "income": 72000,
    "credit_type": "EXP",
    "Credit_Score": 720,
    "term": 360,
    "LTV": 55.5,
    "dtir1": 35.0,
    "approv_in_adv": "nopre",
    "open_credit": "nopc",
    "business_or_commercial": "nob/c",
    "Neg_ammortization": "not_neg",
    "interest_only": "not_int",
    "lump_sum_payment": "not_lpsm",
    "Secured_by": "home",
    "total_units": "1U",
    "Credit_Worthiness": "l1",
    "co_applicant_credit_type": "CIB",
    "submission_of_application": "to_inst",
    "Region": "North",
    "Security_Type": "direct"
  }'
```

**Response:**
```json
{
  "prediction": 0,
  "prediction_label": "No Default",
  "probability_default": 0.4325,
  "probability_no_default": 0.5675,
  "model_version": "rf_balanced_deep_v1",
  "timestamp": "2025-12-19T20:00:00.123456",
  "prediction_id": "pred_a1b2c3d4e5f6"
}
```

### High-Risk Loan Example

**Request:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "loan_limit": "ncf",
    "loan_type": "type3",
    "loan_purpose": "p2",
    "loan_amount": 800000,
    "property_value": 1200000,
    "construction_type": "sb",
    "occupancy_type": "pr",
    "Gender": "Joint",
    "age": "45-54",
    "income": 150000,
    "credit_type": "EQUI",
    "Credit_Score": 650,
    "term": 360,
    "LTV": 66.7,
    "dtir1": 45.0,
    "approv_in_adv": "pre",
    "open_credit": "opc",
    "business_or_commercial": "b/c",
    "Neg_ammortization": "neg_amm",
    "interest_only": "int_only",
    "lump_sum_payment": "lpsm",
    "Secured_by": "land",
    "total_units": "4U",
    "Credit_Worthiness": "l2",
    "co_applicant_credit_type": "EXP",
    "submission_of_application": "not_inst",
    "Region": "North-East",
    "Security_Type": "direct"
  }'
```

**Response:**
```json
{
  "prediction": 1,
  "prediction_label": "Default",
  "probability_default": 0.7234,
  "probability_no_default": 0.2766,
  "model_version": "rf_balanced_deep_v1",
  "timestamp": "2025-12-19T20:05:00.123456",
  "prediction_id": "pred_x7y8z9a0b1c2"
}
```

## Monitoring Drift

### Check Drift Status

**Endpoint:** `GET /drift/status`

**Request:**
```bash
curl http://localhost:8000/drift/status
```

**Response:**
```json
{
  "status": "active",
  "drift_detector_initialized": true,
  "total_predictions_logged": 18,
  "recent_predictions_24h": 16,
  "baseline_samples": 103929,
  "baseline_features": 51,
  "psi_threshold": 0.2,
  "monitoring_active": true
}
```

### Detect Data Drift

**Endpoint:** `POST /drift/detect`

**Request:**
```bash
curl -X POST "http://localhost:8000/drift/detect" \
  -H "Content-Type: application/json" \
  -d '{
    "time_window_hours": 24,
    "features_to_check": null
  }'
```

**Response (abbreviated):**
```json
{
  "timestamp": "2025-12-19T20:10:00.123456",
  "n_samples": 16,
  "summary": {
    "total_features_checked": 51,
    "features_with_drift": 12,
    "drift_percentage": 23.53,
    "overall_drift_detected": true,
    "severity": "medium"
  },
  "features_with_drift": [
    "income",
    "income_to_property",
    "calculated_dti",
    "payment_to_income",
    "LTV",
    "property_value",
    "loan_amount",
    "monthly_payment_est",
    "loan_to_property",
    "Credit_Score",
    "dtir1",
    "term"
  ],
  "detailed_results": [
    {
      "feature": "income",
      "type": "numerical",
      "drift_detected": true,
      "drift_severity": "high",
      "psi": 8.3094,
      "baseline_mean": -0.001,
      "current_mean": 17.5066,
      "mean_change_pct": -1792341.02,
      "baseline_std": 0.9569,
      "current_std": 25.4321
    }
  ]
}
```

### Check Specific Features

**Request:**
```bash
curl -X POST "http://localhost:8000/drift/detect" \
  -H "Content-Type: application/json" \
  -d '{
    "time_window_hours": 1,
    "features_to_check": ["income", "Credit_Score", "LTV"]
  }'
```

## Health Checks

### API Health

**Endpoint:** `GET /health`

**Request:**
```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "rf_balanced_deep_v1",
  "timestamp": "2025-12-19T20:15:00.123456"
}
```

### Root Endpoint

**Endpoint:** `GET /`

**Request:**
```bash
curl http://localhost:8000/
```

**Response:**
```json
{
  "message": "Loan Default Prediction API",
  "version": "1.0.0",
  "endpoints": {
    "health": "/health",
    "predict": "/predict",
    "batch_predict": "/predict/batch",
    "docs": "/docs"
  }
}
```

## Batch Processing

### Batch Predictions

**Endpoint:** `POST /predict/batch`

**Request:**
```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "loans": [
      {
        "loan_limit": "cf",
        "loan_type": "type1",
        "loan_purpose": "p4",
        "loan_amount": 250000,
        "property_value": 450000,
        "construction_type": "sb",
        "occupancy_type": "pr",
        "Gender": "Male",
        "age": "35-44",
        "income": 72000,
        "credit_type": "EXP",
        "Credit_Score": 720,
        "term": 360,
        "LTV": 55.5,
        "dtir1": 35.0,
        "approv_in_adv": "nopre",
        "open_credit": "nopc",
        "business_or_commercial": "nob/c",
        "Neg_ammortization": "not_neg",
        "interest_only": "not_int",
        "lump_sum_payment": "not_lpsm",
        "Secured_by": "home",
        "total_units": "1U",
        "Credit_Worthiness": "l1",
        "co_applicant_credit_type": "CIB",
        "submission_of_application": "to_inst",
        "Region": "North",
        "Security_Type": "direct"
      },
      {
        "loan_limit": "cf",
        "loan_type": "type2",
        "loan_purpose": "p1",
        "loan_amount": 180000,
        "property_value": 350000,
        "construction_type": "sb",
        "occupancy_type": "sr",
        "Gender": "Female",
        "age": "25-34",
        "income": 65000,
        "credit_type": "CRIF",
        "Credit_Score": 680,
        "term": 360,
        "LTV": 51.4,
        "dtir1": 32.0,
        "approv_in_adv": "nopre",
        "open_credit": "nopc",
        "business_or_commercial": "nob/c",
        "Neg_ammortization": "not_neg",
        "interest_only": "not_int",
        "lump_sum_payment": "not_lpsm",
        "Secured_by": "home",
        "total_units": "1U",
        "Credit_Worthiness": "l1",
        "co_applicant_credit_type": "CIB",
        "submission_of_application": "to_inst",
        "Region": "south",
        "Security_Type": "direct"
      }
    ]
  }'
```

**Response:**
```json
{
  "predictions": [
    {
      "prediction": 0,
      "prediction_label": "No Default",
      "probability_default": 0.4325,
      "probability_no_default": 0.5675,
      "model_version": "rf_balanced_deep_v1",
      "timestamp": "2025-12-19T20:20:00.123456",
      "prediction_id": "pred_a1b2c3d4e5f6"
    },
    {
      "prediction": 1,
      "prediction_label": "Default",
      "probability_default": 0.5293,
      "probability_no_default": 0.4707,
      "model_version": "rf_balanced_deep_v1",
      "timestamp": "2025-12-19T20:20:00.234567",
      "prediction_id": "pred_g7h8i9j0k1l2"
    }
  ],
  "total_processed": 2,
  "timestamp": "2025-12-19T20:20:00.345678"
}
```

## Statistics

### Get Prediction Statistics

**Endpoint:** `GET /stats`

**Request:**
```bash
curl http://localhost:8000/stats
```

**Response:**
```json
{
  "total_predictions": 18,
  "predictions_default": 8,
  "predictions_no_default": 10,
  "default_rate": 0.4444,
  "model_version": "rf_balanced_deep_v1"
}
```

## Error Handling

### Model Not Loaded

**Response:**
```json
{
  "detail": "Model not loaded"
}
```
**Status Code:** 503

### Invalid Input

**Response:**
```json
{
  "detail": [
    {
      "loc": ["body", "Credit_Score"],
      "msg": "ensure this value is greater than or equal to 300",
      "type": "value_error.number.not_ge"
    }
  ]
}
```
**Status Code:** 422

### No Recent Predictions for Drift Detection

**Response:**
```json
{
  "detail": "No predictions found in the last 24 hours"
}
```
**Status Code:** 400

## Python Client Example

```python
import requests
import json

BASE_URL = "http://localhost:8000"

# Make a prediction
loan_data = {
    "loan_limit": "cf",
    "loan_type": "type1",
    "loan_purpose": "p4",
    "loan_amount": 250000,
    "property_value": 450000,
    "construction_type": "sb",
    "occupancy_type": "pr",
    "Gender": "Male",
    "age": "35-44",
    "income": 72000,
    "credit_type": "EXP",
    "Credit_Score": 720,
    "term": 360,
    "LTV": 55.5,
    "dtir1": 35.0,
    "approv_in_adv": "nopre",
    "open_credit": "nopc",
    "business_or_commercial": "nob/c",
    "Neg_ammortization": "not_neg",
    "interest_only": "not_int",
    "lump_sum_payment": "not_lpsm",
    "Secured_by": "home",
    "total_units": "1U",
    "Credit_Worthiness": "l1",
    "co_applicant_credit_type": "CIB",
    "submission_of_application": "to_inst",
    "Region": "North",
    "Security_Type": "direct"
}

response = requests.post(f"{BASE_URL}/predict", json=loan_data)
result = response.json()

print(f"Prediction: {result['prediction_label']}")
print(f"Default Probability: {result['probability_default']:.2%}")
print(f"Prediction ID: {result['prediction_id']}")

# Check drift status
drift_status = requests.get(f"{BASE_URL}/drift/status").json()
print(f"\nDrift Monitoring Active: {drift_status['monitoring_active']}")
print(f"Recent Predictions: {drift_status['recent_predictions_24h']}")
```

## Interactive Documentation

Access the full interactive API documentation with try-it-out functionality at:

```
http://localhost:8000/docs
```

Or the alternative ReDoc documentation:

```
http://localhost:8000/redoc
```
