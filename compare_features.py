"""
Compare API-generated features vs training features
"""

import sys
sys.path.insert(0, 'src')

from api.preprocessing import preprocess_for_prediction, load_feature_names
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

# Get expected features
expected_features = load_feature_names()

# Get API-generated features
result = preprocess_for_prediction(test_data, preprocessor)
api_features = list(result.columns)

print("Expected features:", len(expected_features))
print("API features:", len(api_features))

# Check if they match
if set(expected_features) == set(api_features):
    print("\n[SUCCESS] All features present!")

    # Check order
    if expected_features == api_features:
        print("[SUCCESS] Feature order matches exactly!")
    else:
        print("\n[WARNING] Feature order doesn't match!")
        print("\nFirst 10 mismatches:")
        for i in range(min(10, len(expected_features))):
            if expected_features[i] != api_features[i]:
                print(f"  Position {i}: Expected '{expected_features[i]}', got '{api_features[i]}'")
else:
    missing = set(expected_features) - set(api_features)
    extra = set(api_features) - set(expected_features)

    if missing:
        print(f"\n[ERROR] Missing features ({len(missing)}):")
        for feat in list(missing)[:10]:
            print(f"  - {feat}")

    if extra:
        print(f"\n[ERROR] Extra features ({len(extra)}):")
        for feat in list(extra)[:10]:
            print(f"  - {feat}")
