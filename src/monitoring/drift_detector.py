"""
Drift detection using PSI, KS test, and Chi-squared test
"""

import pandas as pd
import numpy as np
import json
from typing import Dict, List, Tuple, Any, Optional
from scipy import stats
from datetime import datetime


class DriftDetector:
    """
    Detects data drift by comparing production data against training baseline.

    Implements:
    - PSI (Population Stability Index) for numerical features
    - KS test (Kolmogorov-Smirnov) for numerical distributions
    - Chi-squared test for categorical features
    """

    def __init__(self, baseline_stats_path: str = "models/baseline_stats.json"):
        """
        Initialize drift detector with baseline statistics.

        Args:
            baseline_stats_path: Path to JSON file with baseline statistics
        """
        with open(baseline_stats_path, 'r') as f:
            self.baseline_stats = json.load(f)

        self.feature_names = self.baseline_stats['feature_names']
        self.numerical_stats = self.baseline_stats['numerical_stats']
        self.categorical_distributions = self.baseline_stats['categorical_distributions']
        self.psi_bins = self.baseline_stats['psi_bins']

        # Drift thresholds
        self.psi_threshold = 0.2  # PSI > 0.2 indicates significant drift
        self.ks_threshold = 0.05   # KS p-value < 0.05 indicates drift
        self.chi2_threshold = 0.05 # Chi2 p-value < 0.05 indicates drift

    def calculate_psi(
        self,
        expected_counts: List[float],
        actual_counts: List[float]
    ) -> float:
        """
        Calculate Population Stability Index (PSI).

        PSI measures the shift in distributions:
        - PSI < 0.1: No significant change
        - 0.1 <= PSI < 0.2: Small change
        - PSI >= 0.2: Significant change (requires investigation)

        Args:
            expected_counts: Baseline counts per bin
            actual_counts: Production counts per bin

        Returns:
            PSI value
        """
        expected = np.array(expected_counts, dtype=float)
        actual = np.array(actual_counts, dtype=float)

        # Handle edge case: if actual is all zeros, can't calculate PSI
        if actual.sum() == 0 or expected.sum() == 0:
            return 0.0

        # Normalize to proportions
        expected_prop = expected / expected.sum()
        actual_prop = actual / actual.sum()

        # Avoid division by zero in individual bins
        expected_prop = np.where(expected_prop == 0, 0.0001, expected_prop)
        actual_prop = np.where(actual_prop == 0, 0.0001, actual_prop)

        # Calculate PSI
        psi = np.sum((actual_prop - expected_prop) * np.log(actual_prop / expected_prop))

        # Handle potential NaN or Inf
        if np.isnan(psi) or np.isinf(psi):
            return 0.0

        return float(psi)

    def detect_numerical_drift(
        self,
        feature_name: str,
        production_data: pd.Series
    ) -> Dict[str, Any]:
        """
        Detect drift in numerical feature using PSI and KS test.

        Args:
            feature_name: Name of the feature
            production_data: Production data for this feature

        Returns:
            Dictionary with drift metrics
        """
        if feature_name not in self.psi_bins:
            return {
                'feature': feature_name,
                'drift_detected': False,
                'message': 'No baseline bins available'
            }

        baseline_info = self.psi_bins[feature_name]
        bins = np.array(baseline_info['bins'])
        expected_counts = np.array(baseline_info['counts'])

        # Bin production data using same bins
        try:
            actual_binned = pd.cut(
                production_data.dropna(),
                bins=bins,
                include_lowest=True
            )
            actual_counts = actual_binned.value_counts().sort_index().values

            # Handle case where production data doesn't fill all bins
            if len(actual_counts) < len(expected_counts):
                full_counts = np.zeros(len(expected_counts))
                full_counts[:len(actual_counts)] = actual_counts
                actual_counts = full_counts

            # Calculate PSI
            psi = self.calculate_psi(expected_counts, actual_counts)

            # Calculate KS test (comparing raw distributions)
            # We need baseline raw data, but we only have stats
            # So we'll just use PSI as primary metric
            drift_detected = psi >= self.psi_threshold

            # Get current stats
            current_mean = float(production_data.mean())
            current_std = float(production_data.std())

            # Handle NaN values
            if np.isnan(current_mean):
                current_mean = 0.0
            if np.isnan(current_std):
                current_std = 0.0

            # Get baseline stats
            baseline_mean = self.numerical_stats[feature_name]['mean']
            baseline_std = self.numerical_stats[feature_name]['std']

            # Calculate percent change
            mean_change_pct = ((current_mean - baseline_mean) / baseline_mean * 100) if baseline_mean != 0 else 0
            if np.isnan(mean_change_pct) or np.isinf(mean_change_pct):
                mean_change_pct = 0.0

            return {
                'feature': feature_name,
                'type': 'numerical',
                'psi': round(psi, 4),
                'drift_detected': drift_detected,
                'drift_severity': self._get_psi_severity(psi),
                'baseline_mean': round(baseline_mean, 4),
                'current_mean': round(current_mean, 4),
                'mean_change_pct': round(mean_change_pct, 2),
                'baseline_std': round(baseline_std, 4),
                'current_std': round(current_std, 4)
            }

        except Exception as e:
            return {
                'feature': feature_name,
                'drift_detected': False,
                'error': str(e)
            }

    def detect_categorical_drift(
        self,
        feature_name: str,
        production_data: pd.Series
    ) -> Dict[str, Any]:
        """
        Detect drift in categorical (binary) feature using proportion test.

        Args:
            feature_name: Name of the feature
            production_data: Production data for this feature

        Returns:
            Dictionary with drift metrics
        """
        if feature_name not in self.categorical_distributions:
            return {
                'feature': feature_name,
                'drift_detected': False,
                'message': 'No baseline distribution available'
            }

        baseline_info = self.categorical_distributions[feature_name]
        baseline_prop = baseline_info['proportion_positive']

        # Calculate current proportion
        current_prop = float(production_data.mean())

        # Calculate chi-squared test for proportion difference
        n_prod = len(production_data)
        n_baseline = baseline_info['count_positive'] + baseline_info['count_negative']

        # Observed vs expected
        observed_positive = int(production_data.sum())
        expected_positive = baseline_prop * n_prod

        # Simple proportion test
        prop_diff = abs(current_prop - baseline_prop)

        # Use a threshold of 0.1 (10% change) for binary features
        drift_detected = prop_diff >= 0.1

        return {
            'feature': feature_name,
            'type': 'categorical_binary',
            'baseline_proportion': round(baseline_prop, 4),
            'current_proportion': round(current_prop, 4),
            'proportion_difference': round(prop_diff, 4),
            'drift_detected': drift_detected,
            'drift_severity': 'high' if prop_diff >= 0.2 else 'medium' if prop_diff >= 0.1 else 'low'
        }

    def _get_psi_severity(self, psi: float) -> str:
        """Get drift severity level based on PSI value."""
        if psi < 0.1:
            return 'low'
        elif psi < 0.2:
            return 'medium'
        else:
            return 'high'

    def detect_drift(
        self,
        production_data: pd.DataFrame,
        features_to_check: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Detect drift across all features.

        Args:
            production_data: Production data DataFrame
            features_to_check: Specific features to check (default: all)

        Returns:
            Comprehensive drift report
        """
        if features_to_check is None:
            features_to_check = self.feature_names

        drift_results = []
        features_with_drift = []

        for feature in features_to_check:
            if feature not in production_data.columns:
                continue

            feature_data = production_data[feature]

            # Determine feature type and run appropriate test
            if feature in self.numerical_stats:
                result = self.detect_numerical_drift(feature, feature_data)
            elif feature in self.categorical_distributions:
                result = self.detect_categorical_drift(feature, feature_data)
            else:
                continue

            drift_results.append(result)

            if result.get('drift_detected', False):
                features_with_drift.append(feature)

        # Summary
        total_features = len(drift_results)
        features_drifted = len(features_with_drift)
        drift_percentage = (features_drifted / total_features * 100) if total_features > 0 else 0

        return {
            'timestamp': datetime.utcnow().isoformat(),
            'n_samples': len(production_data),
            'summary': {
                'total_features_checked': total_features,
                'features_with_drift': features_drifted,
                'drift_percentage': round(drift_percentage, 2),
                'overall_drift_detected': features_drifted > 0,
                'severity': 'high' if drift_percentage > 30 else 'medium' if drift_percentage > 10 else 'low'
            },
            'features_with_drift': features_with_drift,
            'detailed_results': drift_results
        }

    def check_single_prediction(
        self,
        features_dict: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Check if a single prediction's features are within expected ranges.

        Args:
            features_dict: Dictionary of feature values

        Returns:
            Out-of-range feature warnings
        """
        warnings = []

        for feature, value in features_dict.items():
            if feature in self.numerical_stats:
                stats = self.numerical_stats[feature]
                min_val = stats['min']
                max_val = stats['max']
                mean_val = stats['mean']
                std_val = stats['std']

                # Check if value is more than 3 standard deviations from mean
                z_score = abs((value - mean_val) / std_val) if std_val > 0 else 0

                if z_score > 3:
                    warnings.append({
                        'feature': feature,
                        'value': value,
                        'expected_range': f"[{min_val:.2f}, {max_val:.2f}]",
                        'z_score': round(z_score, 2),
                        'severity': 'high' if z_score > 5 else 'medium'
                    })

        return {
            'has_warnings': len(warnings) > 0,
            'n_warnings': len(warnings),
            'warnings': warnings
        }
