"""
Investigate potential data leakage in the raw dataset
"""

import pandas as pd
import numpy as np


def main():
    """Investigate the relationship between missing data and target."""
    print("\n" + "="*60)
    print("DATA LEAKAGE INVESTIGATION")
    print("="*60 + "\n")

    # Load raw data
    print("Loading raw data...")
    df = pd.read_csv('data/raw/Loan_Default.csv')
    print(f"Total rows: {len(df)}")

    # Check complete vs incomplete rows
    complete_rows = df[~df.isnull().any(axis=1)]
    incomplete_rows = df[df.isnull().any(axis=1)]

    print(f"\nComplete rows (no nulls):   {len(complete_rows):6d} ({len(complete_rows)/len(df)*100:.1f}%)")
    print(f"Incomplete rows (has nulls): {len(incomplete_rows):6d} ({len(incomplete_rows)/len(df)*100:.1f}%)")

    # Default rates
    complete_default_rate = complete_rows['Status'].mean()
    incomplete_default_rate = incomplete_rows['Status'].mean()

    print(f"\n{'='*60}")
    print("DEFAULT RATES")
    print("="*60)
    print(f"Complete rows default rate:   {complete_default_rate:.2%}")
    print(f"Incomplete rows default rate: {incomplete_default_rate:.2%}")
    print(f"Difference: {abs(incomplete_default_rate - complete_default_rate):.2%}")

    # Check specific missing indicators
    print(f"\n{'='*60}")
    print("MISSING INDICATOR ANALYSIS")
    print("="*60)

    key_cols = ['rate_of_interest', 'Interest_rate_spread', 'Upfront_charges',
                'property_value', 'LTV', 'income']

    for col in key_cols:
        has_data = df[df[col].notna()]
        missing_data = df[df[col].isna()]

        has_default = has_data['Status'].mean()
        missing_default = missing_data['Status'].mean()

        print(f"\n{col}:")
        print(f"  Has data:  {len(has_data):6d} rows -> {has_default:6.2%} default")
        print(f"  Missing:   {len(missing_data):6d} rows -> {missing_default:6.2%} default")
        print(f"  Gap:       {abs(missing_default - has_default):6.2%}")

    # Check if it's a perfect separator
    print(f"\n{'='*60}")
    print("PERFECT SEPARATOR CHECK")
    print("="*60)

    # Test if missing rate_of_interest alone predicts perfectly
    rate_missing = df['rate_of_interest'].isna()

    # Confusion matrix for rate_of_interest_missing as predictor
    rate_missing_as_pred = rate_missing.astype(int)
    actual = df['Status']

    # Calculate overlap
    correct = (rate_missing_as_pred == actual).sum()
    accuracy_if_used_alone = correct / len(df)

    print(f"\nIf we predicted DEFAULT whenever rate_of_interest is missing:")
    print(f"  Accuracy: {accuracy_if_used_alone:.2%}")

    # Check for suspicious patterns
    print(f"\n{'='*60}")
    print("SUSPICIOUS PATTERN CHECK")
    print("="*60)

    # Do defaults have systematically more missing values?
    defaults = df[df['Status'] == 1]
    non_defaults = df[df['Status'] == 0]

    defaults_null_count = defaults.isnull().sum().sum()
    non_defaults_null_count = non_defaults.isnull().sum().sum()

    avg_nulls_per_default = defaults_null_count / len(defaults)
    avg_nulls_per_non_default = non_defaults_null_count / len(non_defaults)

    print(f"\nAverage null values per row:")
    print(f"  Defaults (Status=1):     {avg_nulls_per_default:.2f} nulls/row")
    print(f"  Non-defaults (Status=0): {avg_nulls_per_non_default:.2f} nulls/row")
    print(f"  Ratio: {avg_nulls_per_default / max(avg_nulls_per_non_default, 0.01):.1f}x more nulls in defaults")

    # Final assessment
    print(f"\n{'='*60}")
    print("ASSESSMENT")
    print("="*60)

    if complete_default_rate < 0.01 and incomplete_default_rate > 0.70:
        print("\n[FINDING] Missing data is a near-perfect predictor of default!")
        print("\nThis could be due to:")
        print("  1. Data collection: Defaults recorded with incomplete info")
        print("  2. Business process: Risky loans have incomplete applications")
        print("  3. Loan type: Certain types (EQUI) both miss data & default more")
        print("\n[RECOMMENDATION]")
        print("  This model will work IF this pattern holds in production.")
        print("  However, monitor carefully - if production data completeness")
        print("  differs, model performance will degrade significantly.")
    else:
        print("\n[FINDING] Missing data is predictive but not deterministic")
        print("  Model appears to have learned valid patterns")


if __name__ == "__main__":
    main()
