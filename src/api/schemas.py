"""
Pydantic schemas for request/response validation
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional
from datetime import datetime


class LoanPredictionRequest(BaseModel):
    """
    Request schema for loan default prediction.
    Contains key features needed for prediction.
    """
    # Loan details
    loan_limit: str = Field(..., description="Loan limit category (cf/ncf)")
    loan_type: str = Field(..., description="Type of loan (type1/type2/type3)")
    loan_purpose: str = Field(..., description="Purpose of loan (p1/p2/p3/p4)")
    loan_amount: float = Field(..., gt=0, description="Loan amount in currency")

    # Property details
    property_value: float = Field(..., gt=0, description="Property value")
    construction_type: str = Field(..., description="Construction type (sb)")
    occupancy_type: str = Field(..., description="Occupancy type (pr/sr/ir)")

    # Applicant details
    Gender: str = Field(..., description="Gender (Male/Female/Joint/Sex Not Available)")
    age: str = Field(..., description="Age group (<25/25-34/35-44/45-54/55-64/65-74/>74)")
    income: float = Field(..., gt=0, description="Annual income")
    credit_type: str = Field(..., description="Credit bureau type (EXP/EQUI/CRIF/CIB)")
    Credit_Score: int = Field(..., ge=300, le=900, description="Credit score")

    # Loan terms
    term: float = Field(..., gt=0, description="Loan term in months")
    LTV: Optional[float] = Field(None, ge=0, le=200, description="Loan-to-value ratio")
    dtir1: Optional[float] = Field(None, ge=0, description="Debt-to-income ratio")

    # Other features
    approv_in_adv: str = Field(..., description="Approval in advance (pre/nopre)")
    open_credit: str = Field(..., description="Open credit (opc/nopc)")
    business_or_commercial: str = Field(..., description="Business or commercial (b/c/nob/c)")
    Neg_ammortization: str = Field(..., description="Negative amortization (neg_amm/not_neg)")
    interest_only: str = Field(..., description="Interest only loan (int_only/not_int)")
    lump_sum_payment: str = Field(..., description="Lump sum payment (lpsm/not_lpsm)")
    Secured_by: str = Field(..., description="Secured by (home)")
    total_units: str = Field(..., description="Total units (1U/2U/3U/4U)")
    Credit_Worthiness: str = Field(..., description="Credit worthiness (l1/l2)")
    co_applicant_credit_type: str = Field(..., description="Co-applicant credit type")
    submission_of_application: str = Field(..., description="Submission method (to_inst/not_inst)")
    Region: str = Field(..., description="Region (south/North/central)")
    Security_Type: str = Field(..., description="Security type (direct)")

    @field_validator('loan_amount', 'property_value', 'income')
    @classmethod
    def validate_positive(cls, v):
        if v <= 0:
            raise ValueError('Must be positive')
        return v

    class Config:
        json_schema_extra = {
            "example": {
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
        }


class LoanPredictionResponse(BaseModel):
    """
    Response schema for loan default prediction.
    """
    prediction: int = Field(..., description="Prediction: 0 (No Default) or 1 (Default)")
    prediction_label: str = Field(..., description="Human-readable prediction")
    probability_default: float = Field(..., ge=0, le=1, description="Probability of default")
    probability_no_default: float = Field(..., ge=0, le=1, description="Probability of no default")
    model_version: str = Field(..., description="Model version used")
    timestamp: datetime = Field(..., description="Prediction timestamp")
    prediction_id: Optional[str] = Field(None, description="Unique prediction ID")

    class Config:
        json_schema_extra = {
            "example": {
                "prediction": 0,
                "prediction_label": "No Default",
                "probability_default": 0.15,
                "probability_no_default": 0.85,
                "model_version": "rf_balanced_deep_v1",
                "timestamp": "2025-12-18T14:30:00",
                "prediction_id": "pred_abc123"
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_version: str
    timestamp: datetime


class BatchPredictionRequest(BaseModel):
    """Batch prediction request."""
    loans: list[LoanPredictionRequest] = Field(..., max_length=100, description="List of loans (max 100)")


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""
    predictions: list[LoanPredictionResponse]
    total_processed: int
    timestamp: datetime


class DriftDetectionRequest(BaseModel):
    """Request for drift detection analysis."""
    time_window_hours: Optional[int] = Field(
        24,
        description="Time window in hours for fetching recent predictions",
        ge=1,
        le=168
    )
    features_to_check: Optional[list[str]] = Field(
        None,
        description="Specific features to check (default: all features)"
    )


class FeatureDriftDetail(BaseModel):
    """Detailed drift information for a single feature."""
    feature: str
    type: str
    drift_detected: bool
    drift_severity: Optional[str] = None
    psi: Optional[float] = None
    baseline_mean: Optional[float] = None
    current_mean: Optional[float] = None
    mean_change_pct: Optional[float] = None
    baseline_std: Optional[float] = None
    current_std: Optional[float] = None
    baseline_proportion: Optional[float] = None
    current_proportion: Optional[float] = None
    proportion_difference: Optional[float] = None
    error: Optional[str] = None


class DriftDetectionResponse(BaseModel):
    """Response from drift detection analysis."""
    timestamp: str
    n_samples: int
    summary: dict
    features_with_drift: list[str]
    detailed_results: list[FeatureDriftDetail]
