"""
Model Training Pipeline with MLflow Experiment Tracking
Trains Random Forest and XGBoost models for loan default prediction
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
    """
    Load preprocessed train and validation data.
    """
    print("Loading preprocessed data...")

    X_train = pd.read_csv('data/processed/X_train.csv')
    y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
    X_val = pd.read_csv('data/processed/X_val.csv')
    y_val = pd.read_csv('data/processed/y_val.csv').values.ravel()

    print(f"Train: {X_train.shape}, Val: {X_val.shape}")
    print(f"Target distribution (train): {np.bincount(y_train)}")

    return X_train, X_val, y_train, y_val


def evaluate_model(model, X_val, y_val):
    """
    Evaluate model and return metrics dictionary.
    """
    # Make predictions
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]

    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred, zero_division=0),
        'recall': recall_score(y_val, y_pred, zero_division=0),
        'f1_score': f1_score(y_val, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_val, y_pred_proba)
    }

    # Get confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    metrics['true_negatives'] = int(cm[0, 0])
    metrics['false_positives'] = int(cm[0, 1])
    metrics['false_negatives'] = int(cm[1, 0])
    metrics['true_positives'] = int(cm[1, 1])

    return metrics


def print_metrics(metrics, model_name="Model"):
    """
    Pretty print evaluation metrics.
    """
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


def train_random_forest(X_train, y_train, X_val, y_val, hyperparameters, experiment_name="loan-default-prediction"):
    """
    Train a Random Forest model with MLflow tracking.

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        hyperparameters: Dict of model hyperparameters
        experiment_name: MLflow experiment name

    Returns:
        Trained model
    """
    print("\n" + "="*60)
    print("TRAINING RANDOM FOREST MODEL")
    print("="*60)

    # Set MLflow experiment
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"RandomForest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Log hyperparameters
        mlflow.log_params(hyperparameters)
        mlflow.log_param("model_type", "RandomForest")

        # Train model
        print("Training model...")
        model = RandomForestClassifier(**hyperparameters)
        model.fit(X_train, y_train)

        # Evaluate on validation set
        print("Evaluating model...")
        metrics = evaluate_model(model, X_val, y_val)

        # Log metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        # Log model
        mlflow.sklearn.log_model(model, "model")

        # Print results
        print_metrics(metrics, "Random Forest")

        # Log feature importance (top 20)
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).head(20)

        print("Top 10 Important Features:")
        for idx, row in feature_importance.head(10).iterrows():
            print(f"  {row['feature']:30s}: {row['importance']:.4f}")

        # Save feature importance as artifact
        importance_path = "feature_importance.csv"
        feature_importance.to_csv(importance_path, index=False)
        mlflow.log_artifact(importance_path)
        os.remove(importance_path)

        return model


def train_xgboost(X_train, y_train, X_val, y_val, hyperparameters, experiment_name="loan-default-prediction"):
    """
    Train an XGBoost model with MLflow tracking.

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        hyperparameters: Dict of model hyperparameters
        experiment_name: MLflow experiment name

    Returns:
        Trained model
    """
    try:
        import xgboost as xgb
    except ImportError:
        print("XGBoost not installed. Skipping XGBoost training.")
        print("Install with: pip install xgboost")
        return None

    print("\n" + "="*60)
    print("TRAINING XGBOOST MODEL")
    print("="*60)

    # Set MLflow experiment
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"XGBoost_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Log hyperparameters
        mlflow.log_params(hyperparameters)
        mlflow.log_param("model_type", "XGBoost")

        # Train model
        print("Training model...")
        model = xgb.XGBClassifier(**hyperparameters)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        # Evaluate on validation set
        print("Evaluating model...")
        metrics = evaluate_model(model, X_val, y_val)

        # Log metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        # Log model
        mlflow.xgboost.log_model(model, "model")

        # Print results
        print_metrics(metrics, "XGBoost")

        # Log feature importance (top 20)
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).head(20)

        print("Top 10 Important Features:")
        for idx, row in feature_importance.head(10).iterrows():
            print(f"  {row['feature']:30s}: {row['importance']:.4f}")

        # Save feature importance as artifact
        importance_path = "feature_importance.csv"
        feature_importance.to_csv(importance_path, index=False)
        mlflow.log_artifact(importance_path)
        os.remove(importance_path)

        return model


def save_best_model(model, model_name, save_dir='models'):
    """
    Save the best model to disk.
    """
    os.makedirs(save_dir, exist_ok=True)
    model_path = f'{save_dir}/{model_name}.pkl'
    joblib.dump(model, model_path)
    print(f"\nBest model saved to: {model_path}")


def main():
    """
    Main training pipeline.
    Trains multiple models and compares performance.
    """
    print("\n" + "="*60)
    print("LOAN DEFAULT PREDICTION - MODEL TRAINING")
    print("="*60 + "\n")

    # Load data
    X_train, X_val, y_train, y_val = load_data()

    # Define experiment name
    experiment_name = "loan-default-prediction"

    # Train Random Forest with different hyperparameters
    rf_configs = [
        {
            'name': 'RF_baseline',
            'params': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'random_state': 42,
                'n_jobs': -1
            }
        },
        {
            'name': 'RF_deeper',
            'params': {
                'n_estimators': 200,
                'max_depth': 15,
                'min_samples_split': 5,
                'random_state': 42,
                'n_jobs': -1
            }
        },
        {
            'name': 'RF_balanced',
            'params': {
                'n_estimators': 150,
                'max_depth': 12,
                'min_samples_split': 10,
                'class_weight': 'balanced',
                'random_state': 42,
                'n_jobs': -1
            }
        }
    ]

    best_model = None
    best_score = 0
    best_name = ""

    # Train Random Forest models
    for config in rf_configs:
        print(f"\nTraining {config['name']}...")
        model = train_random_forest(X_train, y_train, X_val, y_val,
                                    config['params'], experiment_name)

        # Track best model by F1 score
        metrics = evaluate_model(model, X_val, y_val)
        if metrics['f1_score'] > best_score:
            best_score = metrics['f1_score']
            best_model = model
            best_name = config['name']

    # Train XGBoost models
    xgb_configs = [
        {
            'name': 'XGB_baseline',
            'params': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'random_state': 42,
                'n_jobs': -1
            }
        },
        {
            'name': 'XGB_tuned',
            'params': {
                'n_estimators': 200,
                'max_depth': 8,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'n_jobs': -1
            }
        }
    ]

    for config in xgb_configs:
        print(f"\nTraining {config['name']}...")
        model = train_xgboost(X_train, y_train, X_val, y_val,
                             config['params'], experiment_name)

        if model is not None:
            # Track best model by F1 score
            metrics = evaluate_model(model, X_val, y_val)
            if metrics['f1_score'] > best_score:
                best_score = metrics['f1_score']
                best_model = model
                best_name = config['name']

    # Save best model
    if best_model is not None:
        print("\n" + "="*60)
        print(f"BEST MODEL: {best_name} (F1 Score: {best_score:.4f})")
        print("="*60)
        save_best_model(best_model, 'best_model', save_dir='models')

    print("\n[SUCCESS] Training pipeline completed!")
    print("\nTo view results in MLflow UI, run:")
    print("  mlflow ui")
    print("  Then navigate to: http://localhost:5000")


if __name__ == "__main__":
    main()
