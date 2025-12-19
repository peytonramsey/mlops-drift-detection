"""
Model Training with REAL Features (No Missing Indicators)
Trains on actual loan characteristics, not data quality signals
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import joblib
import os
from datetime import datetime


def load_data():
    """Load preprocessed train and validation data (no missing indicators)."""
    print("Loading preprocessed data (NO missing indicators)...")

    X_train = pd.read_csv('data/processed_no_indicators/X_train.csv')
    y_train = pd.read_csv('data/processed_no_indicators/y_train.csv').values.ravel()
    X_val = pd.read_csv('data/processed_no_indicators/X_val.csv')
    y_val = pd.read_csv('data/processed_no_indicators/y_val.csv').values.ravel()

    print(f"Train: {X_train.shape}, Val: {X_val.shape}")
    print(f"Target distribution (train): {np.bincount(y_train)}")
    print(f"Class balance: {np.bincount(y_train)[0]/(np.bincount(y_train)[0]+np.bincount(y_train)[1]):.1%} no default")

    return X_train, X_val, y_train, y_val


def evaluate_model(model, X_val, y_val):
    """Evaluate model and return metrics dictionary."""
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]

    metrics = {
        'accuracy': accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred, zero_division=0),
        'recall': recall_score(y_val, y_pred, zero_division=0),
        'f1_score': f1_score(y_val, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_val, y_pred_proba)
    }

    cm = confusion_matrix(y_val, y_pred)
    metrics['true_negatives'] = int(cm[0, 0])
    metrics['false_positives'] = int(cm[0, 1])
    metrics['false_negatives'] = int(cm[1, 0])
    metrics['true_positives'] = int(cm[1, 1])

    return metrics


def print_metrics(metrics, model_name="Model"):
    """Pretty print evaluation metrics."""
    print(f"\n{'='*60}")
    print(f"{model_name} Performance Metrics")
    print(f"{'='*60}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1_score']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TN: {metrics['true_negatives']:6d}  |  FP: {metrics['false_positives']:6d}")
    print(f"  FN: {metrics['false_negatives']:6d}  |  TP: {metrics['true_positives']:6d}")
    print(f"{'='*60}\n")


def train_random_forest(X_train, y_train, X_val, y_val, hyperparameters, experiment_name="loan-default-real-features"):
    """Train a Random Forest model with real features."""
    print("\n" + "="*60)
    print("TRAINING RANDOM FOREST (REAL FEATURES)")
    print("="*60)

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"RandomForest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        mlflow.log_params(hyperparameters)
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("uses_missing_indicators", False)

        print("Training model...")
        model = RandomForestClassifier(**hyperparameters)
        model.fit(X_train, y_train)

        print("Evaluating model...")
        metrics = evaluate_model(model, X_val, y_val)

        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        mlflow.sklearn.log_model(model, "model")

        print_metrics(metrics, "Random Forest (Real Features)")

        # Log feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).head(20)

        print("Top 10 Important Features:")
        for idx, row in feature_importance.head(10).iterrows():
            print(f"  {row['feature']:30s}: {row['importance']:.4f}")

        importance_path = "feature_importance.csv"
        feature_importance.to_csv(importance_path, index=False)
        mlflow.log_artifact(importance_path)
        os.remove(importance_path)

        return model, metrics


def train_xgboost(X_train, y_train, X_val, y_val, hyperparameters, experiment_name="loan-default-real-features"):
    """Train an XGBoost model with real features."""
    try:
        import xgboost as xgb
    except ImportError:
        print("\nXGBoost not installed. Install with: pip install xgboost")
        return None, None

    print("\n" + "="*60)
    print("TRAINING XGBOOST (REAL FEATURES)")
    print("="*60)

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"XGBoost_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        mlflow.log_params(hyperparameters)
        mlflow.log_param("model_type", "XGBoost")
        mlflow.log_param("uses_missing_indicators", False)

        print("Training model...")
        model = xgb.XGBClassifier(**hyperparameters)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        print("Evaluating model...")
        metrics = evaluate_model(model, X_val, y_val)

        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        mlflow.xgboost.log_model(model, "model")

        print_metrics(metrics, "XGBoost (Real Features)")

        # Log feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).head(20)

        print("Top 10 Important Features:")
        for idx, row in feature_importance.head(10).iterrows():
            print(f"  {row['feature']:30s}: {row['importance']:.4f}")

        importance_path = "feature_importance.csv"
        feature_importance.to_csv(importance_path, index=False)
        mlflow.log_artifact(importance_path)
        os.remove(importance_path)

        return model, metrics


def save_best_model(model, model_name, save_dir='models'):
    """Save the best model to disk."""
    os.makedirs(save_dir, exist_ok=True)
    model_path = f'{save_dir}/{model_name}.pkl'
    joblib.dump(model, model_path)
    print(f"\nBest model saved to: {model_path}")


def main():
    """Main training pipeline for real features."""
    print("\n" + "="*60)
    print("LOAN DEFAULT PREDICTION - REAL FEATURES TRAINING")
    print("="*60 + "\n")

    X_train, X_val, y_train, y_val = load_data()

    experiment_name = "loan-default-real-features"

    # Random Forest configurations with class balancing
    rf_configs = [
        {
            'name': 'RF_balanced',
            'params': {
                'n_estimators': 200,
                'max_depth': 15,
                'min_samples_split': 10,
                'min_samples_leaf': 4,
                'class_weight': 'balanced',
                'random_state': 42,
                'n_jobs': -1
            }
        },
        {
            'name': 'RF_balanced_deep',
            'params': {
                'n_estimators': 300,
                'max_depth': 20,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'class_weight': 'balanced',
                'random_state': 42,
                'n_jobs': -1
            }
        },
        {
            'name': 'RF_weighted',
            'params': {
                'n_estimators': 250,
                'max_depth': 18,
                'min_samples_split': 8,
                'min_samples_leaf': 3,
                'max_features': 'sqrt',
                'class_weight': {0: 1, 1: 3},  # 3x weight for defaults
                'random_state': 42,
                'n_jobs': -1
            }
        }
    ]

    best_model = None
    best_score = 0
    best_name = ""
    all_results = []

    # Train Random Forest models
    for config in rf_configs:
        print(f"\nTraining {config['name']}...")
        model, metrics = train_random_forest(X_train, y_train, X_val, y_val,
                                              config['params'], experiment_name)

        all_results.append({
            'name': config['name'],
            'accuracy': metrics['accuracy'],
            'f1': metrics['f1_score'],
            'roc_auc': metrics['roc_auc']
        })

        if metrics['f1_score'] > best_score:
            best_score = metrics['f1_score']
            best_model = model
            best_name = config['name']

    # Train XGBoost models
    xgb_configs = [
        {
            'name': 'XGB_balanced',
            'params': {
                'n_estimators': 200,
                'max_depth': 8,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'scale_pos_weight': 3,  # Handle class imbalance
                'random_state': 42,
                'n_jobs': -1
            }
        },
        {
            'name': 'XGB_tuned',
            'params': {
                'n_estimators': 300,
                'max_depth': 10,
                'learning_rate': 0.03,
                'subsample': 0.9,
                'colsample_bytree': 0.9,
                'gamma': 0.1,
                'scale_pos_weight': 3,
                'random_state': 42,
                'n_jobs': -1
            }
        }
    ]

    for config in xgb_configs:
        print(f"\nTraining {config['name']}...")
        model, metrics = train_xgboost(X_train, y_train, X_val, y_val,
                                       config['params'], experiment_name)

        if model is not None and metrics is not None:
            all_results.append({
                'name': config['name'],
                'accuracy': metrics['accuracy'],
                'f1': metrics['f1_score'],
                'roc_auc': metrics['roc_auc']
            })

            if metrics['f1_score'] > best_score:
                best_score = metrics['f1_score']
                best_model = model
                best_name = config['name']

    # Print summary
    print("\n" + "="*60)
    print("TRAINING RESULTS SUMMARY")
    print("="*60)
    print(f"{'Model':<25} {'Accuracy':>10} {'F1 Score':>10} {'ROC-AUC':>10}")
    print("-"*60)
    for result in all_results:
        print(f"{result['name']:<25} {result['accuracy']:>10.4f} {result['f1']:>10.4f} {result['roc_auc']:>10.4f}")

    print("\n" + "="*60)
    print(f"BEST MODEL: {best_name}")
    print(f"F1 Score: {best_score:.4f}")
    print("="*60)

    if best_model is not None:
        save_best_model(best_model, 'best_model_real_features', save_dir='models')

    # Final assessment
    print("\n" + "="*60)
    print("ASSESSMENT")
    print("="*60)

    if best_score >= 0.85:
        print("[EXCELLENT] Exceeded 85% F1 score target!")
    elif best_score >= 0.75:
        print("[SUCCESS] Achieved 75-85% performance with real features!")
        print("This is realistic ML performance for this problem.")
    elif best_score >= 0.65:
        print("[ACCEPTABLE] 65-75% performance - consider more feature engineering")
    else:
        print("[REVIEW] Performance below 65% - may need SMOTE or more tuning")

    print("\n[SUCCESS] Training pipeline completed!")
    print("\nTo view results in MLflow UI, run:")
    print("  mlflow ui")
    print("  Then navigate to: http://localhost:5000")


if __name__ == "__main__":
    main()
