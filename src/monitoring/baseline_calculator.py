"""
Calculate baseline statistics from training data for drift detection
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Any


def calculate_baseline_statistics(
    train_data_path: str = "data/processed_no_indicators/X_train.csv",
    output_path: str = "models/baseline_stats.json"
) -> Dict[str, Any]:
    """
    Calculate baseline statistics from training data.

    These statistics will be used to detect drift in production data.

    Args:
        train_data_path: Path to training data CSV
        output_path: Path to save baseline statistics JSON

    Returns:
        Dictionary containing baseline statistics
    """
    print(f"Loading training data from {train_data_path}")
    X_train = pd.read_csv(train_data_path)

    baseline_stats = {
        'n_samples': len(X_train),
        'n_features': len(X_train.columns),
        'feature_names': list(X_train.columns),
        'numerical_stats': {},
        'categorical_distributions': {}
    }

    # Identify numerical vs categorical features
    # Most features are numerical after one-hot encoding
    # But we need to track original categorical distributions
    numerical_features = []

    for col in X_train.columns:
        col_data = X_train[col]

        # Check if column is binary (one-hot encoded categorical)
        unique_vals = col_data.unique()

        if len(unique_vals) == 2 and set(unique_vals).issubset({0, 1}):
            # Binary feature (likely one-hot encoded)
            # Store proportion of 1s
            baseline_stats['categorical_distributions'][col] = {
                'type': 'binary',
                'proportion_positive': float(col_data.mean()),
                'count_positive': int(col_data.sum()),
                'count_negative': int((col_data == 0).sum())
            }
        else:
            # Continuous numerical feature
            numerical_features.append(col)
            baseline_stats['numerical_stats'][col] = {
                'mean': float(col_data.mean()),
                'std': float(col_data.std()),
                'min': float(col_data.min()),
                'max': float(col_data.max()),
                'median': float(col_data.median()),
                'q25': float(col_data.quantile(0.25)),
                'q75': float(col_data.quantile(0.75))
            }

    # Calculate PSI bins for numerical features
    # We'll use 10 bins for PSI calculation
    baseline_stats['psi_bins'] = {}

    for col in numerical_features:
        col_data = X_train[col].dropna()

        # Create 10 quantile-based bins
        try:
            bins = pd.qcut(col_data, q=10, duplicates='drop', retbins=True)[1]
            baseline_stats['psi_bins'][col] = {
                'bins': bins.tolist(),
                'counts': pd.cut(col_data, bins=bins, include_lowest=True).value_counts().sort_index().tolist()
            }
        except Exception as e:
            print(f"Warning: Could not create PSI bins for {col}: {e}")
            # Fallback to equal-width bins
            bins = np.linspace(col_data.min(), col_data.max(), 11)
            baseline_stats['psi_bins'][col] = {
                'bins': bins.tolist(),
                'counts': pd.cut(col_data, bins=bins, include_lowest=True).value_counts().sort_index().tolist()
            }

    # Save baseline statistics
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(baseline_stats, f, indent=2)

    print(f"\nBaseline statistics saved to {output_path}")
    print(f"  - Total samples: {baseline_stats['n_samples']:,}")
    print(f"  - Total features: {baseline_stats['n_features']}")
    print(f"  - Numerical features: {len(numerical_features)}")
    print(f"  - Binary features: {len(baseline_stats['categorical_distributions'])}")

    return baseline_stats


if __name__ == "__main__":
    calculate_baseline_statistics()
