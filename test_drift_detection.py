"""
Test drift detection endpoints
"""

import requests
import json
import time
from datetime import datetime

BASE_URL = "http://localhost:8000"


def test_drift_status():
    """Test drift status endpoint."""
    print("\n" + "="*60)
    print("Testing Drift Status Endpoint")
    print("="*60)

    response = requests.get(f"{BASE_URL}/drift/status")
    print(f"\nStatus Code: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"\nDrift Monitoring Status:")
        print(f"  Status: {data.get('status')}")
        print(f"  Drift Detector Initialized: {data.get('drift_detector_initialized')}")
        print(f"  Total Predictions Logged: {data.get('total_predictions_logged')}")
        print(f"  Recent Predictions (24h): {data.get('recent_predictions_24h')}")
        print(f"  Baseline Samples: {data.get('baseline_samples')}")
        print(f"  Baseline Features: {data.get('baseline_features')}")
        print(f"  PSI Threshold: {data.get('psi_threshold')}")
        print(f"  Monitoring Active: {data.get('monitoring_active')}")
        return True
    else:
        print(f"Error: {response.text}")
        return False


def make_predictions():
    """Make several test predictions to generate production data."""
    print("\n" + "="*60)
    print("Making Test Predictions")
    print("="*60)

    # Normal loans (similar to training data)
    normal_loans = [
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

    # Drifted loans (different from training data - higher amounts, different credit types)
    drifted_loans = [
        {
            "loan_limit": "ncf",
            "loan_type": "type3",
            "loan_purpose": "p2",
            "loan_amount": 800000,  # Much higher than typical
            "property_value": 1200000,  # Much higher
            "construction_type": "sb",
            "occupancy_type": "pr",
            "Gender": "Joint",
            "age": "45-54",
            "income": 150000,  # Higher income
            "credit_type": "EQUI",  # EQUI was high-risk in training
            "Credit_Score": 650,  # Lower credit score
            "term": 360,
            "LTV": 66.7,  # Higher LTV
            "dtir1": 45.0,  # Higher DTI
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
        },
        {
            "loan_limit": "ncf",
            "loan_type": "type3",
            "loan_purpose": "p3",
            "loan_amount": 950000,
            "property_value": 1500000,
            "construction_type": "sb",
            "occupancy_type": "ir",
            "Gender": "Male",
            "age": "55-64",
            "income": 180000,
            "credit_type": "EQUI",
            "Credit_Score": 640,
            "term": 360,
            "LTV": 63.3,
            "dtir1": 42.0,
            "approv_in_adv": "pre",
            "open_credit": "opc",
            "business_or_commercial": "b/c",
            "Neg_ammortization": "not_neg",
            "interest_only": "int_only",
            "lump_sum_payment": "not_lpsm",
            "Secured_by": "home",
            "total_units": "2U",
            "Credit_Worthiness": "l2",
            "co_applicant_credit_type": "EXP",
            "submission_of_application": "to_inst",
            "Region": "central",
            "Security_Type": "direct"
        }
    ]

    all_loans = normal_loans + drifted_loans
    predictions_made = 0

    for i, loan_data in enumerate(all_loans):
        print(f"\nMaking prediction {i+1}/{len(all_loans)}...")
        response = requests.post(f"{BASE_URL}/predict", json=loan_data)

        if response.status_code == 200:
            result = response.json()
            print(f"  Prediction: {result['prediction_label']}")
            print(f"  Default Probability: {result['probability_default']:.2%}")
            predictions_made += 1
        else:
            print(f"  Error: {response.status_code} - {response.text}")

        time.sleep(0.1)  # Small delay between requests

    print(f"\nTotal predictions made: {predictions_made}")
    return predictions_made > 0


def test_drift_detection():
    """Test drift detection endpoint."""
    print("\n" + "="*60)
    print("Testing Drift Detection")
    print("="*60)

    # Test with 1 hour window (should capture all our recent predictions)
    request_data = {
        "time_window_hours": 1,
        "features_to_check": None  # Check all features
    }

    print(f"\nRunning drift detection with {request_data['time_window_hours']} hour window...")

    response = requests.post(f"{BASE_URL}/drift/detect", json=request_data)

    if response.status_code == 200:
        data = response.json()

        print(f"\n[SUCCESS] Drift Detection Complete")
        print(f"\nSummary:")
        print(f"  Samples Analyzed: {data['n_samples']}")
        print(f"  Features Checked: {data['summary']['total_features_checked']}")
        print(f"  Features with Drift: {data['summary']['features_with_drift']}")
        print(f"  Drift Percentage: {data['summary']['drift_percentage']:.2f}%")
        print(f"  Overall Drift Detected: {data['summary']['overall_drift_detected']}")
        print(f"  Severity: {data['summary']['severity']}")

        if data['features_with_drift']:
            print(f"\nTop Features with Drift:")
            # Show up to 10 features with highest drift
            drifted = [r for r in data['detailed_results'] if r.get('drift_detected')]
            drifted_sorted = sorted(
                drifted,
                key=lambda x: x.get('psi', x.get('proportion_difference', 0)),
                reverse=True
            )

            for i, result in enumerate(drifted_sorted[:10]):
                print(f"\n  {i+1}. {result['feature']} ({result['type']})")
                print(f"     Drift Severity: {result.get('drift_severity', 'N/A')}")

                if 'psi' in result and result['psi'] is not None:
                    print(f"     PSI: {result['psi']:.4f}")
                    if result.get('baseline_mean') is not None:
                        print(f"     Mean: {result['baseline_mean']:.4f} -> {result['current_mean']:.4f}")
                        print(f"     Change: {result.get('mean_change_pct', 0):.2f}%")

                if 'proportion_difference' in result and result['proportion_difference'] is not None:
                    print(f"     Proportion Diff: {result['proportion_difference']:.4f}")
                    print(f"     Baseline: {result['baseline_proportion']:.4f}")
                    print(f"     Current: {result['current_proportion']:.4f}")

        return True
    else:
        print(f"\n[ERROR] Drift detection failed")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        return False


def main():
    """Run all drift detection tests."""
    print("\n" + "="*60)
    print("DRIFT DETECTION TEST SUITE")
    print("="*60)

    # Test 1: Check drift status
    print("\n\n--- Test 1: Drift Status ---")
    status_ok = test_drift_status()

    if not status_ok:
        print("\n[FAIL] Drift status endpoint not working")
        return

    # Test 2: Make predictions
    print("\n\n--- Test 2: Generate Production Data ---")
    predictions_ok = make_predictions()

    if not predictions_ok:
        print("\n[FAIL] Could not make predictions")
        return

    # Test 3: Run drift detection
    print("\n\n--- Test 3: Drift Detection ---")
    drift_ok = test_drift_detection()

    if drift_ok:
        print("\n\n" + "="*60)
        print("[SUCCESS] All drift detection tests passed!")
        print("="*60)
    else:
        print("\n\n" + "="*60)
        print("[FAIL] Drift detection test failed")
        print("="*60)


if __name__ == "__main__":
    main()
