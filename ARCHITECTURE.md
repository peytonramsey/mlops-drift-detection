# System Architecture

## Overview

The MLOps Loan Default Prediction System is designed as a production-ready microservice with comprehensive monitoring and drift detection capabilities.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                        Client Layer                          │
│  (Web Apps, Mobile Apps, Other Services, curl, Postman)     │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            │ HTTP/REST
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      API Gateway Layer                       │
│                      (FastAPI Server)                        │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Endpoints:                                             │ │
│  │  • POST /predict          - Single prediction          │ │
│  │  • POST /predict/batch    - Batch predictions          │ │
│  │  • POST /drift/detect     - Drift analysis             │ │
│  │  • GET  /drift/status     - Monitoring status          │ │
│  │  • GET  /health           - Health check               │ │
│  │  • GET  /stats            - Prediction statistics      │ │
│  └────────────────────────────────────────────────────────┘ │
└──────────┬────────────────────────────┬─────────────────────┘
           │                            │
           │ (Pydantic Validation)      │
           ▼                            ▼
┌──────────────────────┐    ┌──────────────────────────┐
│  Preprocessing Layer │    │   Database Layer         │
│                      │    │   (SQLAlchemy + SQLite)  │
│ • Feature Engineering│    │                          │
│ • One-Hot Encoding   │    │  Tables:                 │
│ • Scaling            │    │  • prediction_logs       │
│ • Validation         │    │    - prediction_id       │
└──────────┬───────────┘    │    - timestamp           │
           │                │    - features (JSON)     │
           │                │    - prediction          │
           ▼                │    - probabilities       │
┌──────────────────────┐    │    - model_version       │
│   ML Model Layer     │    └──────────┬───────────────┘
│                      │               │
│ • Random Forest      │               │
│ • 300 estimators     │               │ (Query recent
│ • Balanced weights   │               │  predictions)
│ • 51 features        │               │
└──────────┬───────────┘               │
           │                            │
           │ (Predictions)              ▼
           └──────────────────►┌──────────────────────┐
                                │  Monitoring Layer    │
                                │  (Drift Detector)    │
                                │                      │
                                │ • PSI Calculation    │
                                │ • Statistical Tests  │
                                │ • Baseline Comparison│
                                │ • Alert Generation   │
                                └──────────────────────┘
```

## Component Details

### 1. API Gateway Layer (FastAPI)

**Responsibilities:**
- HTTP request handling
- Input validation with Pydantic
- Response formatting
- CORS middleware
- Error handling
- Swagger/OpenAPI documentation

**Technology:**
- FastAPI 0.100+
- Uvicorn ASGI server
- Pydantic for data validation

**Key Files:**
- `src/api/main.py` - Main application
- `src/api/schemas.py` - Request/response models

### 2. Preprocessing Layer

**Responsibilities:**
- Feature engineering (5 calculated features)
- Categorical encoding (one-hot)
- Feature scaling (StandardScaler)
- Missing value handling

**Pipeline:**
```
Raw Input → Feature Engineering → One-Hot Encoding → Scaling → Model Input
```

**Key Features Created:**
- `calculated_dti` = (loan_amount / income) × 100
- `loan_to_property` = loan_amount / property_value
- `income_to_property` = income / property_value
- `monthly_payment_est` = loan_amount / term
- `payment_to_income` = monthly_payment_est / (income / 12)

**Key Files:**
- `src/api/preprocessing.py`

### 3. ML Model Layer

**Model Specifications:**
- **Algorithm**: Random Forest Classifier
- **Estimators**: 300 trees
- **Max Depth**: 20
- **Class Weight**: Balanced (handles 75/25 class imbalance)
- **Features**: 51 (after preprocessing)
- **Input**: Scaled numerical features
- **Output**: Binary classification (0=No Default, 1=Default)

**Model Artifacts:**
- `models/best_model_real_features.pkl` - Trained model
- `models/scaler_no_indicators.pkl` - StandardScaler
- `models/feature_names.json` - Feature order
- `models/baseline_stats.json` - Baseline statistics

**Key Files:**
- `src/models/train_real_features.py` - Training script
- `src/models/data_preprocessing_no_indicators.py` - Preprocessing class

### 4. Database Layer

**Schema:**
```sql
CREATE TABLE prediction_logs (
    id INTEGER PRIMARY KEY,
    prediction_id TEXT UNIQUE,
    timestamp DATETIME,
    prediction INTEGER,
    probability_default REAL,
    probability_no_default REAL,
    model_version TEXT,
    features JSON,
    -- Key features duplicated for quick queries
    loan_amount REAL,
    property_value REAL,
    income REAL,
    credit_score INTEGER,
    credit_type TEXT,
    loan_type TEXT,
    ltv REAL,
    dtir REAL,
    region TEXT
);
```

**Indexes:**
- `prediction_id` (unique)
- `credit_type`
- `timestamp`

**Key Files:**
- `src/api/database.py`

### 5. Monitoring Layer (Drift Detection)

**Drift Detection Algorithm:**

1. **Fetch Recent Data**: Query predictions from database (configurable time window)
2. **Preprocess**: Apply same transformations as training
3. **Calculate PSI**: For each numerical feature
   ```
   PSI = Σ (Actual% - Expected%) × ln(Actual% / Expected%)
   ```
4. **Classify Drift**:
   - PSI < 0.1: No drift
   - 0.1 ≤ PSI < 0.2: Small drift
   - PSI ≥ 0.2: Significant drift

**Monitored Features:**
- All 51 features (numerical and one-hot encoded)
- Top priority: income, LTV, credit_score, dtir1

**Key Files:**
- `src/monitoring/drift_detector.py` - Drift detection logic
- `src/monitoring/baseline_calculator.py` - Baseline computation

## Data Flow

### Prediction Flow

```
1. Client sends POST /predict with loan data
2. FastAPI validates input (Pydantic)
3. Preprocessing layer transforms features
4. Model generates prediction + probabilities
5. Result logged to database
6. Response returned to client
```

### Drift Detection Flow

```
1. Client sends POST /drift/detect
2. System queries recent predictions from DB
3. For each prediction:
   - Reconstruct preprocessed features
   - Compare to baseline distributions
4. Calculate PSI for each feature
5. Identify drifted features
6. Return comprehensive drift report
```

## Design Decisions

### Why Random Forest?
- Handles non-linear relationships well
- Robust to outliers
- Provides feature importance
- Good performance on tabular data
- Class weights handle imbalance

### Why PSI for Drift Detection?
- Industry standard in financial services
- Interpretable (clear thresholds)
- Works well with binned distributions
- Computationally efficient
- Detects shifts in feature distributions

### Why SQLite?
- Sufficient for prototype/demo
- Zero configuration
- File-based (easy deployment)
- ACID compliance
- Easy to upgrade to PostgreSQL later

### Why FastAPI?
- Automatic OpenAPI documentation
- Fast performance (async)
- Type validation with Pydantic
- Modern Python syntax
- Easy testing

## Scalability Considerations

### Current Limitations
- SQLite (single-writer bottleneck)
- In-memory model loading
- Single-process server

### Production Enhancements
1. **Database**: Migrate to PostgreSQL
2. **Caching**: Add Redis for frequently accessed data
3. **Load Balancing**: Deploy multiple API instances behind Nginx
4. **Message Queue**: Use Celery for async drift detection
5. **Monitoring**: Integrate Prometheus + Grafana
6. **Logging**: Structured JSON logs to ELK stack

## Security Considerations

### Current Implementation
- No authentication (prototype)
- CORS enabled (development)
- Input validation (Pydantic)

### Production Requirements
- [ ] API key authentication
- [ ] Rate limiting (per client)
- [ ] HTTPS/TLS encryption
- [ ] Input sanitization
- [ ] SQL injection prevention (using ORM)
- [ ] Secrets management (environment variables)
- [ ] Audit logging

## Performance

### Response Times (Typical)
- Single prediction: ~50-100ms
- Batch prediction (10 loans): ~200-300ms
- Drift detection (100 predictions): ~500-1000ms
- Health check: <10ms

### Throughput
- Single instance: ~100-200 requests/second
- With load balancing: Linearly scalable

## Monitoring Metrics

### Model Metrics
- Prediction latency (p50, p95, p99)
- Predictions per second
- Default rate over time
- Model version in use

### Drift Metrics
- Number of features with drift
- Average PSI across features
- Drift severity distribution
- Time since last drift alert

### System Metrics
- API uptime
- Database connection pool
- Memory usage
- CPU utilization

## Deployment Architecture

### Docker Deployment
```
docker-compose up
├── API Container (port 8000)
│   ├── FastAPI app
│   ├── Model artifacts
│   └── Database (mounted volume)
```

### Cloud Deployment (Future)
```
Cloud Load Balancer
├── API Instance 1 (ECS/Cloud Run)
├── API Instance 2 (ECS/Cloud Run)
├── API Instance N (ECS/Cloud Run)
└── Shared Database (RDS/Cloud SQL)
    ├── Prediction logs
    └── Drift history
```

## Future Enhancements

1. **Real-time Alerts**: Email/Slack notifications on drift
2. **A/B Testing**: Compare multiple model versions
3. **Auto-retraining**: Trigger retraining on drift
4. **Feature Store**: Centralized feature management
5. **Model Registry**: MLflow for model versioning
6. **Explainability**: SHAP values for predictions
7. **Dashboard**: React/Vue.js frontend
