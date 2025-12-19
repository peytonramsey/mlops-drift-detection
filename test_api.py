"""
Test script for FastAPI loan default prediction
"""

import requests
import json


# API base URL
BASE_URL = "http://localhost:8000"


def test_health():
    """Test health check endpoint."""
    print("\n" + "="*60)
    print("Testing Health Endpoint")
    print("="*60)

    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    return response.status_code == 200


def test_prediction():
    """Test prediction endpoint with sample data."""
    print("\n" + "="*60)
    print("Testing Prediction Endpoint")
    print("="*60)

    # Sample loan application (good loan - should not default)
    good_loan = {
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

    print("\nTesting GOOD loan (expected: No Default):")
    print(f"  - Loan Amount: ${good_loan['loan_amount']:,}")
    print(f"  - Income: ${good_loan['income']:,}")
    print(f"  - Credit Score: {good_loan['Credit_Score']}")
    print(f"  - LTV: {good_loan['LTV']}%")

    response = requests.post(f"{BASE_URL}/predict", json=good_loan)
    print(f"\nStatus Code: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        print(f"\nPrediction: {result['prediction_label']}")
        print(f"Probability of Default: {result['probability_default']:.2%}")
        print(f"Probability of No Default: {result['probability_no_default']:.2%}")
        print(f"Model Version: {result['model_version']}")
        print(f"Prediction ID: {result['prediction_id']}")
    else:
        print(f"Error: {response.text}")

    return response.status_code == 200


def test_risky_loan():
    """Test with a risky loan (EQUI credit type)."""
    print("\n" + "="*60)
    print("Testing Risky Loan")
    print("="*60)

    # Risky loan - EQUI credit type, high LTV, lower credit score
    risky_loan = {
        "loan_limit": "cf",
        "loan_type": "type3",
        "loan_purpose": "p3",
        "loan_amount": 350000,
        "property_value": 380000,
        "construction_type": "sb",
        "occupancy_type": "pr",
        "Gender": "Male",
        "age": "25-34",
        "income": 45000,
        "credit_type": "EQUI",  # EQUI is high risk
        "Credit_Score": 580,  # Lower score
        "term": 360,
        "LTV": 92.0,  # High LTV
        "dtir1": 48.0,  # High debt-to-income
        "approv_in_adv": "nopre",
        "open_credit": "opc",
        "business_or_commercial": "nob/c",
        "Neg_ammortization": "not_neg",
        "interest_only": "not_int",
        "lump_sum_payment": "not_lpsm",
        "Secured_by": "home",
        "total_units": "1U",
        "Credit_Worthiness": "l2",
        "co_applicant_credit_type": "EQUI",
        "submission_of_application": "to_inst",
        "Region": "south",
        "Security_Type": "direct"
    }

    print("\nTesting RISKY loan (expected: Default):")
    print(f"  - Loan Amount: ${risky_loan['loan_amount']:,}")
    print(f"  - Income: ${risky_loan['income']:,}")
    print(f"  - Credit Score: {risky_loan['Credit_Score']}")
    print(f"  - Credit Type: {risky_loan['credit_type']} (high risk)")
    print(f"  - LTV: {risky_loan['LTV']}%")

    response = requests.post(f"{BASE_URL}/predict", json=risky_loan)
    print(f"\nStatus Code: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        print(f"\nPrediction: {result['prediction_label']}")
        print(f"Probability of Default: {result['probability_default']:.2%}")
        print(f"Probability of No Default: {result['probability_no_default']:.2%}")
    else:
        print(f"Error: {response.text}")

    return response.status_code == 200


def test_stats():
    """Test statistics endpoint."""
    print("\n" + "="*60)
    print("Testing Statistics Endpoint")
    print("="*60)

    response = requests.get(f"{BASE_URL}/stats")
    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        stats = response.json()
        print(f"\nPrediction Statistics:")
        print(f"  Total Predictions: {stats['total_predictions']}")
        print(f"  Predicted Defaults: {stats['predictions_default']}")
        print(f"  Predicted No Defaults: {stats['predictions_no_default']}")
        print(f"  Default Rate: {stats['default_rate']:.2%}")
    else:
        print(f"Error: {response.text}")

    return response.status_code == 200


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("LOAN DEFAULT PREDICTION API - TEST SUITE")
    print("="*60)

    try:
        # Test health
        if not test_health():
            print("\n[ERROR] Health check failed!")
            return

        # Test predictions
        test_prediction()
        test_risky_loan()

        # Test stats
        test_stats()

        print("\n" + "="*60)
        print("[SUCCESS] All tests completed!")
        print("="*60)

    except requests.exceptions.ConnectionError:
        print("\n[ERROR] Could not connect to API. Is it running?")
        print("Start the API with: uvicorn src.api.main:app --reload")
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")


if __name__ == "__main__":
    main()
