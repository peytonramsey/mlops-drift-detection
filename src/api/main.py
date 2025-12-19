"""
FastAPI application for loan default prediction
"""

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import uuid
import os

from .schemas import (
    LoanPredictionRequest,
    LoanPredictionResponse,
    HealthResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    DriftDetectionRequest,
    DriftDetectionResponse,
    FeatureDriftDetail
)
from .database import get_db, init_db, PredictionLog
from .preprocessing import preprocess_for_prediction
from src.monitoring.drift_detector import DriftDetector

# Initialize FastAPI app
app = FastAPI(
    title="Loan Default Prediction API",
    description="Production ML API for predicting loan defaults with drift monitoring",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model and preprocessor
MODEL = None
PREPROCESSOR = None
MODEL_VERSION = "rf_balanced_deep_v1"
DRIFT_DETECTOR = None


def load_model_and_preprocessor():
    """Load trained model and preprocessor artifacts."""
    global MODEL, PREPROCESSOR

    try:
        # Load model
        model_path = "models/best_model_real_features.pkl"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")

        MODEL = joblib.load(model_path)
        print(f"Model loaded from {model_path}")

        # Load preprocessor artifacts
        PREPROCESSOR = {
            'scaler': joblib.load('models/scaler_no_indicators.pkl'),
            'numerical_medians': joblib.load('models/numerical_medians_no_indicators.pkl'),
            'categorical_modes': joblib.load('models/categorical_modes_no_indicators.pkl')
        }
        print("Preprocessor artifacts loaded")

        return True
    except Exception as e:
        print(f"Error loading model/preprocessor: {e}")
        return False


@app.on_event("startup")
async def startup_event():
    """Initialize database and load model on startup."""
    global DRIFT_DETECTOR

    init_db()
    success = load_model_and_preprocessor()
    if not success:
        print("WARNING: Model/Preprocessor failed to load!")

    # Initialize drift detector
    try:
        DRIFT_DETECTOR = DriftDetector()
        print("Drift detector initialized")
    except Exception as e:
        print(f"WARNING: Drift detector failed to initialize: {e}")


@app.get("/", response_model=dict)
def root():
    """Root endpoint."""
    return {
        "message": "Loan Default Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "batch_predict": "/predict/batch",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if MODEL is not None else "unhealthy",
        model_loaded=MODEL is not None,
        model_version=MODEL_VERSION,
        timestamp=datetime.utcnow()
    )




@app.post("/predict", response_model=LoanPredictionResponse)
def predict(
    request: LoanPredictionRequest,
    db: Session = Depends(get_db)
):
    """
    Make a loan default prediction.

    Args:
        request: Loan application details
        db: Database session

    Returns:
        Prediction result with probabilities
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Preprocess input
        input_data = request.model_dump()
        X = preprocess_for_prediction(input_data, PREPROCESSOR)

        # Make prediction
        prediction = int(MODEL.predict(X)[0])
        probabilities = MODEL.predict_proba(X)[0]

        prob_no_default = float(probabilities[0])
        prob_default = float(probabilities[1])

        # Generate prediction ID
        prediction_id = f"pred_{uuid.uuid4().hex[:12]}"

        # Create response
        response = LoanPredictionResponse(
            prediction=prediction,
            prediction_label="Default" if prediction == 1 else "No Default",
            probability_default=prob_default,
            probability_no_default=prob_no_default,
            model_version=MODEL_VERSION,
            timestamp=datetime.utcnow(),
            prediction_id=prediction_id
        )

        # Log to database
        log_entry = PredictionLog(
            prediction_id=prediction_id,
            timestamp=response.timestamp,
            prediction=prediction,
            probability_default=prob_default,
            probability_no_default=prob_no_default,
            model_version=MODEL_VERSION,
            features=input_data,
            loan_amount=request.loan_amount,
            property_value=request.property_value,
            income=request.income,
            credit_score=request.Credit_Score,
            credit_type=request.credit_type,
            loan_type=request.loan_type,
            ltv=request.LTV,
            dtir=request.dtir1,
            region=request.Region
        )

        db.add(log_entry)
        db.commit()

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def batch_predict(
    request: BatchPredictionRequest,
    db: Session = Depends(get_db)
):
    """
    Make predictions for multiple loans at once.

    Args:
        request: Batch of loan applications
        db: Database session

    Returns:
        Batch prediction results
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    predictions = []
    for loan_request in request.loans:
        try:
            prediction = predict(loan_request, db)
            predictions.append(prediction)
        except Exception as e:
            print(f"Error processing loan: {e}")
            continue

    return BatchPredictionResponse(
        predictions=predictions,
        total_processed=len(predictions),
        timestamp=datetime.utcnow()
    )


@app.get("/stats")
def get_prediction_stats(db: Session = Depends(get_db)):
    """Get statistics about predictions made."""
    total_predictions = db.query(PredictionLog).count()
    predictions_default = db.query(PredictionLog).filter(PredictionLog.prediction == 1).count()
    predictions_no_default = db.query(PredictionLog).filter(PredictionLog.prediction == 0).count()

    return {
        "total_predictions": total_predictions,
        "predictions_default": predictions_default,
        "predictions_no_default": predictions_no_default,
        "default_rate": predictions_default / total_predictions if total_predictions > 0 else 0,
        "model_version": MODEL_VERSION
    }


@app.post("/drift/detect", response_model=DriftDetectionResponse)
def detect_drift(
    request: DriftDetectionRequest = DriftDetectionRequest(),
    db: Session = Depends(get_db)
):
    """
    Detect data drift in recent predictions.

    Analyzes recent predictions to detect distribution shifts compared to training data.

    Args:
        request: Drift detection parameters
        db: Database session

    Returns:
        Drift detection report with PSI and statistical tests
    """
    if DRIFT_DETECTOR is None:
        raise HTTPException(status_code=503, detail="Drift detector not initialized")

    try:
        # Fetch recent predictions from database
        from datetime import timedelta
        time_cutoff = datetime.utcnow() - timedelta(hours=request.time_window_hours)

        recent_predictions = db.query(PredictionLog).filter(
            PredictionLog.timestamp >= time_cutoff
        ).all()

        if len(recent_predictions) == 0:
            raise HTTPException(
                status_code=400,
                detail=f"No predictions found in the last {request.time_window_hours} hours"
            )

        # Extract features from logged predictions
        # We need to reconstruct the preprocessed features
        production_data_list = []

        for pred in recent_predictions:
            # The 'features' field contains the raw input
            # We need to preprocess it the same way we do for predictions
            try:
                from .preprocessing import preprocess_for_prediction

                preprocessed = preprocess_for_prediction(pred.features, PREPROCESSOR)
                production_data_list.append(preprocessed.iloc[0])
            except Exception as e:
                print(f"Error preprocessing logged prediction: {e}")
                continue

        if len(production_data_list) == 0:
            raise HTTPException(
                status_code=500,
                detail="Failed to preprocess any logged predictions"
            )

        production_data = pd.DataFrame(production_data_list)

        # Run drift detection
        drift_report = DRIFT_DETECTOR.detect_drift(
            production_data,
            features_to_check=request.features_to_check
        )

        # Convert to response model
        detailed_results = [
            FeatureDriftDetail(**result)
            for result in drift_report['detailed_results']
        ]

        return DriftDetectionResponse(
            timestamp=drift_report['timestamp'],
            n_samples=drift_report['n_samples'],
            summary=drift_report['summary'],
            features_with_drift=drift_report['features_with_drift'],
            detailed_results=detailed_results
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Drift detection failed: {str(e)}")


@app.get("/drift/status")
def get_drift_status(db: Session = Depends(get_db)):
    """
    Get current drift monitoring status.

    Returns:
        Drift monitoring statistics and recent drift alerts
    """
    if DRIFT_DETECTOR is None:
        return {
            "status": "not_initialized",
            "message": "Drift detector not initialized"
        }

    # Get total predictions
    total_predictions = db.query(PredictionLog).count()

    # Get recent predictions (last 24 hours)
    from datetime import timedelta
    time_cutoff = datetime.utcnow() - timedelta(hours=24)

    recent_count = db.query(PredictionLog).filter(
        PredictionLog.timestamp >= time_cutoff
    ).count()

    return {
        "status": "active",
        "drift_detector_initialized": True,
        "total_predictions_logged": total_predictions,
        "recent_predictions_24h": recent_count,
        "baseline_samples": DRIFT_DETECTOR.baseline_stats['n_samples'],
        "baseline_features": DRIFT_DETECTOR.baseline_stats['n_features'],
        "psi_threshold": DRIFT_DETECTOR.psi_threshold,
        "monitoring_active": recent_count > 0
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
