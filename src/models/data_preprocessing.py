"""
Data Preprocessing Pipeline for Loan Default Prediction
Handles missing values, feature engineering, encoding, and train/test splits
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os


class LoanDataPreprocessor:
    """
    Preprocessing pipeline for loan default dataset.
    Handles missing values with indicators, encoding, and scaling.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.categorical_mappings = {}
        self.numerical_medians = {}
        self.categorical_modes = {}

    def create_missing_indicators(self, df):
        """
        Create binary indicators for missing values.
        CRITICAL: Missing data is highly predictive (72.58% default rate)!
        """
        print("Creating missing value indicators...")

        missing_indicator_cols = [
            'property_value', 'LTV', 'rate_of_interest',
            'Interest_rate_spread', 'Upfront_charges',
            'dtir1', 'income'
        ]

        for col in missing_indicator_cols:
            if col in df.columns:
                df[f'{col}_missing'] = df[col].isna().astype(int)
                print(f"  - {col}_missing: {df[f'{col}_missing'].sum()} rows")

        return df

    def handle_income_zero(self, df):
        """
        Handle income = 0 (99.37% default rate!).
        Treat as missing data since it's likely a data quality issue.
        """
        print("Handling income = 0...")

        zero_count = (df['income'] == 0).sum()
        if zero_count > 0:
            df['income_is_zero'] = (df['income'] == 0).astype(int)
            df['income'] = df['income'].replace(0, np.nan)
            print(f"  - Found {zero_count} rows with income=0, created indicator")

        return df

    def drop_high_missing_columns(self, df):
        """
        Drop columns with >25% missing values that are less critical.
        """
        print("Dropping high-missing columns...")

        cols_to_drop = ['Upfront_charges']
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
            'rate_of_interest', 'Interest_rate_spread',
            'dtir1', 'property_value', 'LTV', 'income',
            'loan_amount', 'term', 'age', 'Credit_Score', 'total_units'
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
        Drop rows with remaining nulls (should be <0.5% of data).
        """
        initial_rows = len(df)
        df = df.dropna()
        dropped = initial_rows - len(df)

        if dropped > 0:
            print(f"Dropped {dropped} rows with remaining nulls ({dropped/initial_rows*100:.2f}%)")

        return df

    def encode_categorical(self, df, fit=True):
        """
        One-hot encode categorical variables.
        """
        print("Encoding categorical features...")

        # Identify categorical columns (exclude target and ID)
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

        # Remove ID and Status if present
        categorical_cols = [col for col in categorical_cols if col not in ['ID', 'Status']]

        if categorical_cols:
            df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)
            print(f"  - Encoded {len(categorical_cols)} categorical columns")

        return df

    def scale_features(self, df, target_col='Status', fit=True):
        """
        Scale numerical features using StandardScaler.
        Note: Not strictly necessary for tree-based models but good practice.
        """
        print("Scaling numerical features...")

        # Separate features and target
        if target_col in df.columns:
            X = df.drop(columns=[target_col, 'ID'], errors='ignore')
            y = df[target_col]
        else:
            X = df.drop(columns=['ID'], errors='ignore')
            y = None

        # Get numerical columns (excluding binary indicators)
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        binary_cols = [col for col in numerical_cols if col.endswith('_missing') or col == 'income_is_zero']
        cols_to_scale = [col for col in numerical_cols if col not in binary_cols]

        if cols_to_scale:
            if fit:
                X[cols_to_scale] = self.scaler.fit_transform(X[cols_to_scale])
            else:
                X[cols_to_scale] = self.scaler.transform(X[cols_to_scale])

            print(f"  - Scaled {len(cols_to_scale)} numerical columns")

        # Reconstruct dataframe
        if y is not None:
            df = pd.concat([X, y], axis=1)
        else:
            df = X

        return df

    def preprocess(self, df, fit=True, scale=True, target_col='Status'):
        """
        Main preprocessing pipeline.

        Args:
            df: Input dataframe
            fit: Whether to fit transformers (True for train, False for test)
            scale: Whether to scale features
            target_col: Name of target column

        Returns:
            Preprocessed dataframe
        """
        print("\n" + "="*60)
        print("STARTING PREPROCESSING PIPELINE")
        print("="*60)
        print(f"Initial shape: {df.shape}")

        # Create a copy to avoid modifying original
        df = df.copy()

        # Step 1: Create missing indicators (BEFORE imputation!)
        df = self.create_missing_indicators(df)

        # Step 2: Handle income = 0
        df = self.handle_income_zero(df)

        # Step 3: Drop high-missing columns
        df = self.drop_high_missing_columns(df)

        # Step 4: Impute numerical features
        df = self.impute_numerical(df, fit=fit)

        # Step 5: Impute categorical features
        df = self.impute_categorical(df, fit=fit)

        # Step 6: Drop remaining nulls
        df = self.drop_remaining_nulls(df)

        # Step 7: Encode categorical variables
        df = self.encode_categorical(df, fit=fit)

        # Step 8: Scale features (optional)
        if scale:
            df = self.scale_features(df, target_col=target_col, fit=fit)

        print(f"\nFinal shape: {df.shape}")
        print("="*60)
        print("PREPROCESSING COMPLETE")
        print("="*60 + "\n")

        return df

    def save_artifacts(self, save_dir='models'):
        """
        Save preprocessing artifacts (scaler, medians, modes).
        """
        os.makedirs(save_dir, exist_ok=True)

        joblib.dump(self.scaler, f'{save_dir}/scaler.pkl')
        joblib.dump(self.numerical_medians, f'{save_dir}/numerical_medians.pkl')
        joblib.dump(self.categorical_modes, f'{save_dir}/categorical_modes.pkl')

        print(f"Preprocessing artifacts saved to {save_dir}/")

    def load_artifacts(self, save_dir='models'):
        """
        Load preprocessing artifacts.
        """
        self.scaler = joblib.load(f'{save_dir}/scaler.pkl')
        self.numerical_medians = joblib.load(f'{save_dir}/numerical_medians.pkl')
        self.categorical_modes = joblib.load(f'{save_dir}/categorical_modes.pkl')

        print(f"Preprocessing artifacts loaded from {save_dir}/")


def split_data(df, target_col='Status', test_size=0.2, val_size=0.1, random_state=42):
    """
    Split data into train/validation/test sets with stratification.

    Args:
        df: Preprocessed dataframe
        target_col: Name of target column
        test_size: Proportion for test set (default 0.2 = 20%)
        val_size: Proportion for validation set (default 0.1 = 10%)
        random_state: Random seed for reproducibility

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    print("\n" + "="*60)
    print("SPLITTING DATA")
    print("="*60)

    # Separate features and target
    X = df.drop(columns=[target_col, 'ID'], errors='ignore')
    y = df[target_col]

    print(f"Total samples: {len(X)}")
    print(f"Features: {X.shape[1]}")
    print(f"Target distribution:\n{y.value_counts(normalize=True)}")

    # Split into train+val and test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Split train+val into train and val
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
    Example usage of preprocessing pipeline.
    """
    # Load raw data
    print("Loading raw data...")
    df = pd.read_csv('data/raw/Loan_Default.csv')
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # Initialize preprocessor
    preprocessor = LoanDataPreprocessor()

    # Preprocess data
    df_processed = preprocessor.preprocess(df, fit=True, scale=True)

    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_processed)

    # Save processed data
    print("Saving processed data...")
    os.makedirs('data/processed', exist_ok=True)

    X_train.to_csv('data/processed/X_train.csv', index=False)
    X_val.to_csv('data/processed/X_val.csv', index=False)
    X_test.to_csv('data/processed/X_test.csv', index=False)
    y_train.to_csv('data/processed/y_train.csv', index=False)
    y_val.to_csv('data/processed/y_val.csv', index=False)
    y_test.to_csv('data/processed/y_test.csv', index=False)

    print("Processed data saved to data/processed/")

    # Save preprocessing artifacts
    preprocessor.save_artifacts('models')

    print("\n[SUCCESS] Preprocessing pipeline completed successfully!")


if __name__ == "__main__":
    main()
