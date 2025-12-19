"""
Helper module for preprocessing API inputs to match training data format
"""

import pandas as pd
import numpy as np
import json


def load_feature_names():
    """Load the feature names from training."""
    with open('models/feature_names.json', 'r') as f:
        return json.load(f)


def preprocess_for_prediction(input_data: dict, preprocessor: dict) -> pd.DataFrame:
    """
    Preprocess raw input to match training data format exactly.

    Args:
        input_data: Dict with raw feature values
        preprocessor: Dict with scaler and medians

    Returns:
        DataFrame with features matching training format
    """
    # Load expected features
    expected_features = load_feature_names()

    # Start with raw data
    df = pd.DataFrame([input_data])

    # Add year (constant for now)
    df['year'] = 2019

    # Handle missing LTV/dtir
    if 'LTV' not in df.columns or pd.isna(df['LTV'].iloc[0]) or df['LTV'].iloc[0] is None:
        df['LTV'] = (df['loan_amount'] / df['property_value']) * 100

    if 'dtir1' not in df.columns or pd.isna(df['dtir1'].iloc[0]) or df['dtir1'].iloc[0] is None:
        df['dtir1'] = preprocessor['numerical_medians'].get('dtir1', 39.0)

    # Create engineered features
    df['calculated_dti'] = (df['loan_amount'] / df['income']) * 100
    df['loan_to_property'] = df['loan_amount'] / df['property_value']
    df['income_to_property'] = df['income'] / df['property_value']
    df['monthly_payment_est'] = df['loan_amount'] / df['term']
    df['payment_to_income'] = df['monthly_payment_est'] / (df['income'] / 12)

    # One-hot encode categorical variables (must match training exactly!)
    categorical_cols = [
        'loan_limit', 'Gender', 'age', 'approv_in_adv', 'loan_type', 'loan_purpose',
        'Credit_Worthiness', 'open_credit', 'business_or_commercial',
        'Neg_ammortization', 'interest_only', 'lump_sum_payment',
        'construction_type', 'occupancy_type', 'Secured_by', 'total_units',
        'credit_type', 'co_applicant_credit_type', 'submission_of_application',
        'Region', 'Security_Type'
    ]

    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)

    # Create a dataframe with all expected features initialized to 0
    result_df = pd.DataFrame(0, index=[0], columns=expected_features)

    # Fill in the values we have
    for col in df_encoded.columns:
        if col in expected_features:
            result_df[col] = df_encoded[col].values[0]

    # Scale ALL features (scaler was fit on all 51 features during training)
    if preprocessor and 'scaler' in preprocessor:
        try:
            # Transform all features in the same order as training
            result_df_scaled = preprocessor['scaler'].transform(result_df)
            # Convert back to DataFrame with column names
            result_df = pd.DataFrame(result_df_scaled, columns=result_df.columns, index=result_df.index)
        except Exception as e:
            print(f"Warning: Scaling failed: {e}")

    return result_df
