"""
Test preprocessing function directly
"""

import sys
sys.path.insert(0, 'src')

from api.preprocessing import preprocess_for_prediction
import joblib

# Load preprocessor
preprocessor = {
    'scaler': joblib.load('models/scaler_no_indicators.pkl'),
    'numerical_medians': joblib.load('models/numerical_medians_no_indicators.pkl'),
    'categorical_modes': joblib.load('models/categorical_modes_no_indicators.pkl')
}

# Test data
test_data = {
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

print("Testing preprocessing...")
try:
    result = preprocess_for_prediction(test_data, preprocessor)
    print(f"\nResult shape: {result.shape}")
    print(f"\nFeatures ({len(result.columns)}):")
    print(list(result.columns[:20]))

    # Check if age columns are present
    age_cols = [col for col in result.columns if 'age' in col.lower()]
    print(f"\nAge-related columns found: {age_cols}")

    # Check if "age" (non-encoded) is still there
    if 'age' in result.columns:
        print("\n[ERROR] Raw 'age' column still present - not properly encoded!")
    else:
        print("\n[SUCCESS] Raw 'age' column removed - properly one-hot encoded")

except Exception as e:
    print(f"\n[ERROR] Preprocessing failed: {e}")
    import traceback
    traceback.print_exc()
