"""
Data Preprocessing Pipeline WITHOUT Missing Indicators
Forces model to learn from actual features, not data quality signals
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os


class LoanDataPreprocessorNoIndicators:
    """
    Preprocessing pipeline WITHOUT missing indicators.
    This forces the model to learn from real features.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.numerical_medians = {}
        self.categorical_modes = {}

    def handle_income_zero(self, df):
        """
        Handle income = 0 by treating as missing (but don't create indicator).
        """
        print("Handling income = 0...")
        zero_count = (df['income'] == 0).sum()
        if zero_count > 0:
            df['income'] = df['income'].replace(0, np.nan)
            print(f"  - Found {zero_count} rows with income=0, treating as NaN")
        return df

    def drop_high_missing_columns(self, df):
        """
        Drop columns with >25% missing values.
        """
        print("Dropping high-missing columns...")

        cols_to_drop = ['Upfront_charges', 'rate_of_interest', 'Interest_rate_spread']
        existing_cols = [col for col in cols_to_drop if col in df.columns]

        if existing_cols:
            df = df.drop(columns=existing_cols)
            print(f"  - Dropped: {existing_cols}")

        return df

    def impute_numerical(self, df, fit=True):
        """
        Impute numerical features with median.
        """
        print("Imputing numerical features...")

        numerical_cols = [
            'dtir1', 'property_value', 'LTV', 'income', 'loan_amount',
            'term', 'age', 'Credit_Score', 'total_units', 'loan_limit'
        ]

        for col in numerical_cols:
            if col in df.columns and df[col].dtype in [np.float64, np.int64] and df[col].isna().any():
                if fit:
                    self.numerical_medians[col] = df[col].median()

                fill_value = self.numerical_medians.get(col, df[col].median())
                df.loc[:, col] = df[col].fillna(fill_value)
                print(f"  - {col}: filled with {fill_value:.2f}")

        return df

    def impute_categorical(self, df, fit=True):
        """
        Impute categorical features with mode.
        """
        print("Imputing categorical features...")

        categorical_cols = [
            'approv_in_adv', 'loan_purpose', 'Neg_ammortization',
            'Gender', 'loan_limit', 'loan_type', 'submission_of_application'
        ]

        for col in categorical_cols:
            if col in df.columns and df[col].isna().any():
                if fit:
                    mode_value = df[col].mode()
                    self.categorical_modes[col] = mode_value[0] if len(mode_value) > 0 else 'Unknown'

                fill_value = self.categorical_modes.get(col, 'Unknown')
                df.loc[:, col] = df[col].fillna(fill_value)
                print(f"  - {col}: filled with '{fill_value}'")

        return df

    def drop_remaining_nulls(self, df):
        """
        Drop rows with remaining nulls.
        """
        initial_rows = len(df)
        df = df.dropna()
        dropped = initial_rows - len(df)

        if dropped > 0:
            print(f"Dropped {dropped} rows with remaining nulls ({dropped/initial_rows*100:.2f}%)")

        return df

    def create_engineered_features(self, df):
        """
        Create useful engineered features.
        """
        print("Creating engineered features...")

        # Debt to income ratio (if not already present and dtir1 is missing)
        if 'dtir1' in df.columns and 'loan_amount' in df.columns and 'income' in df.columns:
            # Calculate our own DTI where it might be missing
            calculated_dti = (df['loan_amount'] / df['income'].replace(0, np.nan)) * 100
            df['calculated_dti'] = calculated_dti
            print("  - Created calculated_dti")

        # Loan to property value ratio
        if 'loan_amount' in df.columns and 'property_value' in df.columns:
            df['loan_to_property'] = df['loan_amount'] / df['property_value'].replace(0, 1)
            print("  - Created loan_to_property")

        # Income to property ratio
        if 'income' in df.columns and 'property_value' in df.columns:
            df['income_to_property'] = df['income'] / df['property_value'].replace(0, 1)
            print("  - Created income_to_property")

        # Monthly payment estimate (approximate)
        if 'loan_amount' in df.columns and 'term' in df.columns:
            df['monthly_payment_est'] = df['loan_amount'] / df['term'].replace(0, 1)
            print("  - Created monthly_payment_est")

        # Payment to income ratio
        if 'monthly_payment_est' in df.columns and 'income' in df.columns:
            df['payment_to_income'] = df['monthly_payment_est'] / (df['income'].replace(0, 1) / 12)
            print("  - Created payment_to_income")

        return df

    def encode_categorical(self, df, fit=True):
        """
        One-hot encode categorical variables.
        """
        print("Encoding categorical features...")

        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        categorical_cols = [col for col in categorical_cols if col not in ['ID', 'Status']]

        if categorical_cols:
            df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)
            print(f"  - Encoded {len(categorical_cols)} categorical columns")

        return df

    def scale_features(self, df, target_col='Status', fit=True):
        """
        Scale numerical features using StandardScaler.
        """
        print("Scaling numerical features...")

        if target_col in df.columns:
            X = df.drop(columns=[target_col, 'ID'], errors='ignore')
            y = df[target_col]
        else:
            X = df.drop(columns=['ID'], errors='ignore')
            y = None

        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()

        if numerical_cols:
            if fit:
                X[numerical_cols] = self.scaler.fit_transform(X[numerical_cols])
            else:
                X[numerical_cols] = self.scaler.transform(X[numerical_cols])

            print(f"  - Scaled {len(numerical_cols)} numerical columns")

        if y is not None:
            df = pd.concat([X, y], axis=1)
        else:
            df = X

        return df

    def preprocess(self, df, fit=True, scale=True, target_col='Status'):
        """
        Main preprocessing pipeline WITHOUT missing indicators.
        """
        print("\n" + "="*60)
        print("PREPROCESSING PIPELINE (NO MISSING INDICATORS)")
        print("="*60)
        print(f"Initial shape: {df.shape}")

        df = df.copy()

        # Step 1: Handle income = 0
        df = self.handle_income_zero(df)

        # Step 2: Drop high-missing columns
        df = self.drop_high_missing_columns(df)

        # Step 3: Impute numerical features
        df = self.impute_numerical(df, fit=fit)

        # Step 4: Impute categorical features
        df = self.impute_categorical(df, fit=fit)

        # Step 5: Create engineered features
        df = self.create_engineered_features(df)

        # Step 6: Drop remaining nulls
        df = self.drop_remaining_nulls(df)

        # Step 7: Encode categorical variables
        df = self.encode_categorical(df, fit=fit)

        # Step 8: Scale features
        if scale:
            df = self.scale_features(df, target_col=target_col, fit=fit)

        print(f"\nFinal shape: {df.shape}")
        print("="*60)
        print("PREPROCESSING COMPLETE (NO MISSING INDICATORS)")
        print("="*60 + "\n")

        return df

    def save_artifacts(self, save_dir='models'):
        """
        Save preprocessing artifacts.
        """
        os.makedirs(save_dir, exist_ok=True)

        joblib.dump(self.scaler, f'{save_dir}/scaler_no_indicators.pkl')
        joblib.dump(self.numerical_medians, f'{save_dir}/numerical_medians_no_indicators.pkl')
        joblib.dump(self.categorical_modes, f'{save_dir}/categorical_modes_no_indicators.pkl')

        print(f"Preprocessing artifacts saved to {save_dir}/")

    def load_artifacts(self, save_dir='models'):
        """
        Load preprocessing artifacts.
        """
        self.scaler = joblib.load(f'{save_dir}/scaler_no_indicators.pkl')
        self.numerical_medians = joblib.load(f'{save_dir}/numerical_medians_no_indicators.pkl')
        self.categorical_modes = joblib.load(f'{save_dir}/categorical_modes_no_indicators.pkl')

        print(f"Preprocessing artifacts loaded from {save_dir}/")


def split_data(df, target_col='Status', test_size=0.2, val_size=0.1, random_state=42):
    """
    Split data into train/validation/test sets with stratification.
    """
    print("\n" + "="*60)
    print("SPLITTING DATA")
    print("="*60)

    X = df.drop(columns=[target_col, 'ID'], errors='ignore')
    y = df[target_col]

    print(f"Total samples: {len(X)}")
    print(f"Features: {X.shape[1]}")
    print(f"Target distribution:\n{y.value_counts(normalize=True)}")

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=y_temp
    )

    print(f"\nTrain set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"Val set:   {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"Test set:  {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
    print("="*60 + "\n")

    return X_train, X_val, X_test, y_train, y_val, y_test


def main():
    """
    Run preprocessing pipeline without missing indicators.
    """
    print("Loading raw data...")
    df = pd.read_csv('data/raw/Loan_Default.csv')
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    preprocessor = LoanDataPreprocessorNoIndicators()
    df_processed = preprocessor.preprocess(df, fit=True, scale=True)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_processed)

    print("Saving processed data...")
    os.makedirs('data/processed_no_indicators', exist_ok=True)

    X_train.to_csv('data/processed_no_indicators/X_train.csv', index=False)
    X_val.to_csv('data/processed_no_indicators/X_val.csv', index=False)
    X_test.to_csv('data/processed_no_indicators/X_test.csv', index=False)
    y_train.to_csv('data/processed_no_indicators/y_train.csv', index=False)
    y_val.to_csv('data/processed_no_indicators/y_val.csv', index=False)
    y_test.to_csv('data/processed_no_indicators/y_test.csv', index=False)

    print("Processed data saved to data/processed_no_indicators/")

    preprocessor.save_artifacts('models')

    print("\n[SUCCESS] Preprocessing pipeline completed (no missing indicators)!")


if __name__ == "__main__":
    main()
