"""
Evaluate trained model on held-out test set
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns


def load_test_data():
    """Load test data."""
    print("Loading test data...")
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()
    print(f"Test set: {X_test.shape}")
    print(f"Target distribution: {np.bincount(y_test)}")
    return X_test, y_test


def evaluate_model(model, X_test, y_test):
    """Comprehensive model evaluation."""
    print("\n" + "="*60)
    print("EVALUATING MODEL ON TEST SET")
    print("="*60)

    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Print results
    print("\n" + "="*60)
    print("TEST SET PERFORMANCE")
    print("="*60)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1_score']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TN: {cm[0,0]:6d}  |  FP: {cm[0,1]:6d}")
    print(f"  FN: {cm[1,0]:6d}  |  TP: {cm[1,1]:6d}")
    print("="*60)

    # Detailed classification report
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=['No Default (0)', 'Default (1)']))

    # Check for data leakage indicators
    print("\n" + "="*60)
    print("DATA LEAKAGE CHECK")
    print("="*60)

    if metrics['accuracy'] >= 0.99:
        print("[WARNING] Accuracy >= 99% - Possible data leakage!")
        print("  Check if any features contain target information")
    elif metrics['accuracy'] >= 0.95:
        print("[INFO] Accuracy >= 95% - Very high performance")
        print("  Review top features to ensure they're valid predictors")
    else:
        print("[OK] Performance seems reasonable for real-world data")

    # Analyze predictions
    print(f"\nPrediction Distribution:")
    print(f"  Predicted 0 (No Default): {np.sum(y_pred == 0)} ({np.sum(y_pred == 0)/len(y_pred)*100:.1f}%)")
    print(f"  Predicted 1 (Default):    {np.sum(y_pred == 1)} ({np.sum(y_pred == 1)/len(y_pred)*100:.1f}%)")
    print(f"\nActual Distribution:")
    print(f"  Actual 0 (No Default):    {np.sum(y_test == 0)} ({np.sum(y_test == 0)/len(y_test)*100:.1f}%)")
    print(f"  Actual 1 (Default):       {np.sum(y_test == 1)} ({np.sum(y_test == 1)/len(y_test)*100:.1f}%)")

    return metrics, cm


def check_feature_importance(model, X_test):
    """Analyze top features for data leakage."""
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*60)

    feature_importance = pd.DataFrame({
        'feature': X_test.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nTop 15 Most Important Features:")
    for idx, row in feature_importance.head(15).iterrows():
        print(f"  {row['feature']:35s}: {row['importance']:.4f}")

    # Highlight missing indicators
    missing_indicators = feature_importance[feature_importance['feature'].str.contains('_missing')]
    if not missing_indicators.empty:
        print(f"\nMissing Indicators Total Importance: {missing_indicators['importance'].sum():.2%}")
        print("Top 5 Missing Indicators:")
        for idx, row in missing_indicators.head(5).iterrows():
            print(f"  {row['feature']:35s}: {row['importance']:.4f}")


def main():
    """Main evaluation pipeline."""
    print("\n" + "="*60)
    print("TEST SET EVALUATION")
    print("="*60 + "\n")

    # Load model
    print("Loading best model...")
    model = joblib.load('models/best_model.pkl')
    print(f"Model loaded: {type(model).__name__}")

    # Load test data
    X_test, y_test = load_test_data()

    # Evaluate
    metrics, cm = evaluate_model(model, X_test, y_test)

    # Feature importance
    check_feature_importance(model, X_test)

    # Final verdict
    print("\n" + "="*60)
    print("FINAL VERDICT")
    print("="*60)

    if metrics['accuracy'] >= 0.99:
        print("[INVESTIGATE] Perfect/near-perfect performance detected!")
        print("Action items:")
        print("  1. Review preprocessing for data leakage")
        print("  2. Check if ID column or similar was accidentally included")
        print("  3. Verify train/test split was done correctly")
        print("  4. Ensure target variable isn't encoded in features")
    elif metrics['accuracy'] >= 0.85:
        print("[SUCCESS] Model meets Week 1 goal (>85% accuracy)!")
        print("Ready to proceed to Week 2 (API deployment)")
    else:
        print("[REVIEW NEEDED] Model performance below 85% target")
        print("Consider:")
        print("  - Hyperparameter tuning")
        print("  - Feature engineering")
        print("  - Trying XGBoost or other algorithms")

    print("\n[COMPLETE] Test evaluation finished!")


if __name__ == "__main__":
    main()
