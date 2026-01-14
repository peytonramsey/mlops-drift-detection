---
title: Loan Default Prediction System
emoji: 🏦
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
---

# 🏦 Loan Default Prediction System

Production-grade ML system for predicting loan defaults with real-time drift detection and monitoring.

## 🎯 Live Demo

Try the interactive demo above! Enter loan details to get instant default risk predictions.

## 📊 Model Performance

- **Accuracy**: 88.93%
- **F1 Score**: 0.86
- **ROC-AUC**: 0.92
- **Dataset**: 148,670 loan applications
- **Features**: 51 engineered features
- **Inference**: <100ms latency

## 🛠️ Tech Stack

- **ML Framework**: Random Forest, XGBoost, PyTorch
- **API**: FastAPI
- **Monitoring**: PSI-based drift detection
- **Experiment Tracking**: MLflow
- **Database**: SQLAlchemy
- **Deployment**: Docker

## 🔍 Key Features

- **Real-time Predictions**: Sub-100ms inference latency
- **Drift Detection**: Multi-method monitoring (PSI, KS test, Chi-squared)
- **Production Ready**: Dockerized, fully tested, production-grade code
- **Interactive UI**: Clean web interface for easy testing
- **API Documentation**: Full OpenAPI/Swagger docs at `/docs`

## 📈 API Endpoints

- `POST /predict` - Make a single prediction
- `POST /predict/batch` - Batch predictions
- `GET /drift/status` - Check drift monitoring status
- `POST /drift/detect` - Detect data drift
- `GET /health` - Health check

## 🚀 Usage

### Web Interface
Simply use the form above to test predictions!

### API (cURL)
```bash
curl -X POST "https://your-space.hf.space/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "loan_amount": 250000,
    "property_value": 450000,
    "income": 72000,
    "Credit_Score": 720,
    "LTV": 55.5,
    "dtir1": 35.0,
    "term": 360,
    "age": "35-44",
    "Gender": "Male",
    ...
  }'
```

### Python
```python
import requests

data = {
    "loan_amount": 250000,
    "property_value": 450000,
    "income": 72000,
    "Credit_Score": 720,
    # ... other fields
}

response = requests.post("https://your-space.hf.space/predict", json=data)
print(response.json())
```

## 🎓 About This Project

Built as a portfolio project demonstrating end-to-end MLOps practices:

- ✅ Feature engineering (DTI, LTV ratios)
- ✅ Model training & comparison (Random Forest vs XGBoost)
- ✅ Production API deployment
- ✅ Real-time monitoring & drift detection
- ✅ MLflow experiment tracking
- ✅ Docker containerization
- ✅ Clean UI for demos

## 📧 Contact

**Peyton Ramsey**
- Email: ramseypeyton@gmail.com
- GitHub: [@peytonramsey](https://github.com/peytonramsey)
- Project: [mlops-drift-detection](https://github.com/peytonramsey/mlops-drift-detection)

## 📄 License

MIT License - feel free to use this project as a reference!

---

Built with ❤️ for ML Engineer roles
