# MLOps Loan Default Prediction System

A production-grade machine learning system for predicting loan defaults with real-time drift detection and monitoring capabilities.

## Overview

This project demonstrates end-to-end MLOps practices by building a loan default prediction API with automated data drift detection. The system monitors production data distributions and alerts when model performance may degrade due to distributional shifts.

## Key Features

- **Production ML API**: FastAPI-based REST API for real-time loan default predictions
- **Drift Detection**: PSI (Population Stability Index) monitoring for 51 features
- **Automated Logging**: SQLite database tracking all predictions and features
- **Interactive Documentation**: Swagger UI for API exploration
- **Model Performance**: 88.93% accuracy, 73.52% F1 score on test data
- **Batch Predictions**: Support for processing multiple loan applications
- **Health Monitoring**: System status and drift monitoring endpoints

## Tech Stack

- **ML Framework**: scikit-learn (Random Forest with balanced class weights)
- **API**: FastAPI, Pydantic, Uvicorn
- **Database**: SQLAlchemy with SQLite
- **Monitoring**: Custom PSI-based drift detector
- **Data Processing**: pandas, numpy
- **Deployment**: Docker, Docker Compose

## Architecture

```
┌─────────────────┐
│  Client Request │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│         FastAPI Server              │
│  ┌───────────────────────────────┐  │
│  │  /predict - Make Prediction   │  │
│  │  /drift/detect - Check Drift  │  │
│  │  /drift/status - Monitor      │  │
│  │  /health - Health Check       │  │
│  └───────────────────────────────┘  │
└──────────┬──────────────────────────┘
           │
    ┌──────┴──────┐
    │             │
    ▼             ▼
┌────────┐   ┌──────────────┐
│ Model  │   │   Database   │
│  .pkl  │   │  (SQLite)    │
└────────┘   └──────────────┘
    │             │
    │             ▼
    │        ┌──────────────┐
    │        │ Drift        │
    │        │ Detector     │
    │        └──────────────┘
    │             │
    └─────────────┘
```

## Project Structure

```
mlops-drift-detection/
├── src/
│   ├── api/
│   │   ├── main.py              # FastAPI application
│   │   ├── schemas.py           # Pydantic models
│   │   ├── database.py          # SQLAlchemy models
│   │   └── preprocessing.py     # Feature engineering
│   ├── models/
│   │   ├── data_preprocessing_no_indicators.py
│   │   └── train_real_features.py
│   └── monitoring/
│       ├── drift_detector.py    # PSI calculation & drift detection
│       └── baseline_calculator.py
├── models/
│   ├── best_model_real_features.pkl
│   ├── scaler_no_indicators.pkl
│   ├── baseline_stats.json
│   └── feature_names.json
├── data/
│   └── processed_no_indicators/
│       ├── X_train.csv
│       └── X_test.csv
├── notebooks/
│   └── eda.ipynb               # Exploratory Data Analysis
├── test_api.py                 # API integration tests
├── test_drift_detection.py     # Drift detection tests
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Quick Start

### Prerequisites

- Python 3.9+
- pip

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/mlops-drift-detection.git
cd mlops-drift-detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the API:
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

4. Access the interactive API documentation:
```
http://localhost:8000/docs
```

### Using Docker (Recommended)

1. Build and run with Docker Compose:
```bash
docker-compose up --build
```

2. Access the API at `http://localhost:8000`

## API Usage

### Make a Prediction

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
  "probability_default": 0.15,
  "probability_no_default": 0.85,
  "model_version": "rf_balanced_deep_v1",
  "timestamp": "2025-12-19T20:00:00",
  "prediction_id": "pred_abc123def456"
}
```

### Check Drift Status

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

```bash
curl -X POST "http://localhost:8000/drift/detect" \
  -H "Content-Type: application/json" \
  -d '{"time_window_hours": 24}'
```

**Response includes:**
- Number of samples analyzed
- Features with detected drift
- PSI values for each feature
- Drift severity (low/medium/high)
- Distribution statistics

## Model Details

### Training Data
- **Dataset**: 148,670 loan applications
- **Features**: 51 (after preprocessing and feature engineering)
- **Target**: Binary classification (Default / No Default)
- **Class Balance**: 75% No Default, 25% Default

### Model Performance
- **Algorithm**: Random Forest (300 estimators, max_depth=20)
- **Accuracy**: 88.93%
- **Precision**: 73.85%
- **Recall**: 73.19%
- **F1 Score**: 73.52%
- **ROC-AUC**: 89.08%

### Top Features by Importance
1. credit_type_EQUI (17.4%)
2. dtir1 (10.7%)
3. LTV (10.4%)
4. Credit_Score (8.7%)
5. income (7.5%)

## Drift Detection

The system uses **PSI (Population Stability Index)** to detect distributional shifts:

- **PSI < 0.1**: No significant change
- **0.1 ≤ PSI < 0.2**: Small change
- **PSI ≥ 0.2**: Significant change (requires investigation)

When drift is detected in critical features, the model may need retraining to maintain accuracy.

## Testing

Run the test suite:

```bash
# Test API endpoints
python test_api.py

# Test drift detection
python test_drift_detection.py
```

## Development

### Train the Model

```bash
python src/models/train_real_features.py
```

### Calculate Baseline Statistics

```bash
python src/monitoring/baseline_calculator.py
```

### Run with Auto-Reload (Development)

```bash
uvicorn src.api.main:app --reload
```

## Production Considerations

- **Database**: Replace SQLite with PostgreSQL for production
- **Authentication**: Add API key or OAuth authentication
- **Rate Limiting**: Implement request throttling
- **Monitoring**: Integrate with Prometheus/Grafana
- **Logging**: Use structured logging (e.g., JSON logs)
- **Alerts**: Set up email/Slack notifications for drift
- **Scaling**: Use Kubernetes for horizontal scaling

## Future Enhancements

- [ ] A/B testing framework for model comparison
- [ ] Automated model retraining pipeline
- [ ] Grafana dashboard for real-time monitoring
- [ ] Performance degradation detection
- [ ] Model explainability (SHAP values)
- [ ] Cloud deployment (AWS/GCP/Azure)

## Acknowledgments

- Dataset: Loan Default Dataset from [source]
- Inspiration: Production MLOps best practices
- Built with: FastAPI, scikit-learn, and open-source tools
