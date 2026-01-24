"""
PyTorch Data Loading Pipeline for Loan Default Prediction
Replaces pandas/sklearn data loading with PyTorch Dataset and DataLoader
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import joblib
import os
from typing import Optional, Tuple, Dict, Any


class LoanDefaultDataset(Dataset):
    """
    PyTorch Dataset for Loan Default Prediction.

    Handles data loading, preprocessing, and transformations in PyTorch format.
    Compatible with existing preprocessing pipeline.
    """

    def __init__(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        transform: Optional[Any] = None,
        scaler: Optional[Any] = None,
        device: str = 'cpu'
    ):
        """
        Initialize dataset.

        Args:
            X: Feature dataframe (preprocessed)
            y: Target series (optional, for prediction mode)
            transform: Optional transformations to apply
            scaler: Optional scaler for normalization
            device: Device to load tensors to ('cpu' or 'cuda')
        """
        self.device = device
        self.transform = transform
        self.scaler = scaler

        # Convert to tensors
        self.X = torch.FloatTensor(X.values).to(device)

        if y is not None:
            self.y = torch.LongTensor(y.values).to(device)
        else:
            self.y = None

        # Store feature names for later use
        self.feature_names = X.columns.tolist()
        self.n_features = len(self.feature_names)

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.

        Args:
            idx: Index of sample

        Returns:
            Tuple of (features, target) if y exists, else just features
        """
        x = self.X[idx]

        # Apply transformations if provided
        if self.transform:
            x = self.transform(x)

        # Return with or without target
        if self.y is not None:
            return x, self.y[idx]
        else:
            return x

    def get_feature_names(self) -> list:
        """Get list of feature names."""
        return self.feature_names

    def get_sample_weights(self) -> torch.Tensor:
        """
        Calculate sample weights for imbalanced classes.
        Useful for weighted loss functions.
        """
        if self.y is None:
            raise ValueError("Cannot calculate sample weights without target labels")

        # Count class occurrences
        class_counts = torch.bincount(self.y)

        # Calculate weights (inverse frequency)
        weights = 1.0 / class_counts.float()
        weights = weights / weights.sum()  # Normalize

        # Map weights to samples
        sample_weights = torch.zeros(len(self.y))
        for i, label in enumerate(self.y):
            sample_weights[i] = weights[label]

        return sample_weights


class PyTorchPreprocessor:
    """
    PyTorch-compatible preprocessing pipeline.

    This wraps the existing sklearn preprocessing but provides
    PyTorch-friendly interfaces and tensor operations.
    """

    def __init__(self):
        """Initialize preprocessor."""
        self.scaler = None
        self.numerical_medians = {}
        self.categorical_modes = {}
        self.feature_names = None
        self.n_features = None

    def load_artifacts(self, save_dir: str = 'models'):
        """
        Load preprocessing artifacts from sklearn pipeline.

        Args:
            save_dir: Directory containing artifacts
        """
        scaler_path = f'{save_dir}/scaler.pkl'
        medians_path = f'{save_dir}/numerical_medians.pkl'
        modes_path = f'{save_dir}/categorical_modes.pkl'

        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            print(f"Loaded scaler from {scaler_path}")

        if os.path.exists(medians_path):
            self.numerical_medians = joblib.load(medians_path)
            print(f"Loaded medians from {medians_path}")

        if os.path.exists(modes_path):
            self.categorical_modes = joblib.load(modes_path)
            print(f"Loaded modes from {modes_path}")

    def create_dataloaders(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[pd.Series] = None,
        batch_size: int = 32,
        shuffle_train: bool = True,
        num_workers: int = 0,
        device: str = 'cpu'
    ) -> Dict[str, DataLoader]:
        """
        Create PyTorch DataLoaders from preprocessed data.

        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            X_test, y_test: Optional test data
            batch_size: Batch size for training
            shuffle_train: Whether to shuffle training data
            num_workers: Number of worker processes for data loading
            device: Device to load data to

        Returns:
            Dictionary with 'train', 'val', and optionally 'test' DataLoaders
        """
        # Create datasets
        train_dataset = LoanDefaultDataset(X_train, y_train, device=device)
        val_dataset = LoanDefaultDataset(X_val, y_val, device=device)

        # Store feature info
        self.feature_names = train_dataset.get_feature_names()
        self.n_features = train_dataset.n_features

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
            pin_memory=(device == 'cuda')
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device == 'cuda')
        )

        loaders = {
            'train': train_loader,
            'val': val_loader
        }

        # Add test loader if provided
        if X_test is not None and y_test is not None:
            test_dataset = LoanDefaultDataset(X_test, y_test, device=device)
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=(device == 'cuda')
            )
            loaders['test'] = test_loader

        print(f"\nDataLoaders created:")
        print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
        print(f"  Val:   {len(val_dataset)} samples, {len(val_loader)} batches")
        if 'test' in loaders:
            print(f"  Test:  {len(test_dataset)} samples, {len(loaders['test'])} batches")
        print(f"  Batch size: {batch_size}")
        print(f"  Features: {self.n_features}")

        return loaders


class TensorScaler:
    """
    PyTorch-native scaler for normalization.

    This is a PyTorch alternative to sklearn's StandardScaler,
    using learned parameters for normalization.
    """

    def __init__(self, n_features: int):
        """
        Initialize scaler.

        Args:
            n_features: Number of features to scale
        """
        self.n_features = n_features
        self.mean = None
        self.std = None
        self.fitted = False

    def fit(self, X: torch.Tensor):
        """
        Fit scaler to data.

        Args:
            X: Input tensor of shape (n_samples, n_features)
        """
        self.mean = X.mean(dim=0, keepdim=True)
        self.std = X.std(dim=0, keepdim=True)

        # Avoid division by zero
        self.std[self.std == 0] = 1.0

        self.fitted = True

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """
        Transform data using fitted parameters.

        Args:
            X: Input tensor

        Returns:
            Scaled tensor
        """
        if not self.fitted:
            raise RuntimeError("Scaler must be fitted before transform")

        return (X - self.mean) / self.std

    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X: torch.Tensor) -> torch.Tensor:
        """
        Inverse transform scaled data back to original scale.

        Args:
            X: Scaled tensor

        Returns:
            Original scale tensor
        """
        if not self.fitted:
            raise RuntimeError("Scaler must be fitted before inverse transform")

        return X * self.std + self.mean


def load_preprocessed_data(
    data_dir: str = 'data/processed',
    device: str = 'cpu'
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load preprocessed CSV files.

    Args:
        data_dir: Directory containing preprocessed CSVs
        device: Device for tensors

    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    print(f"\nLoading preprocessed data from {data_dir}...")

    X_train = pd.read_csv(f'{data_dir}/X_train.csv')
    y_train = pd.read_csv(f'{data_dir}/y_train.csv').values.ravel()
    y_train = pd.Series(y_train)

    X_val = pd.read_csv(f'{data_dir}/X_val.csv')
    y_val = pd.read_csv(f'{data_dir}/y_val.csv').values.ravel()
    y_val = pd.Series(y_val)

    X_test = pd.read_csv(f'{data_dir}/X_test.csv')
    y_test = pd.read_csv(f'{data_dir}/y_test.csv').values.ravel()
    y_test = pd.Series(y_test)

    print(f"  Train: {X_train.shape}")
    print(f"  Val:   {X_val.shape}")
    print(f"  Test:  {X_test.shape}")
    print(f"  Target distribution (train): {np.bincount(y_train)}")

    return X_train, y_train, X_val, y_val, X_test, y_test


def get_class_weights(y: pd.Series, device: str = 'cpu') -> torch.Tensor:
    """
    Calculate class weights for imbalanced datasets.

    Args:
        y: Target labels
        device: Device for tensor

    Returns:
        Tensor of class weights
    """
    class_counts = np.bincount(y)
    weights = 1.0 / class_counts
    weights = weights / weights.sum()  # Normalize

    return torch.FloatTensor(weights).to(device)


def main():
    """
    Example usage of PyTorch data loading pipeline.
    """
    print("\n" + "="*70)
    print("PYTORCH DATA LOADING PIPELINE - DEMO")
    print("="*70)

    # Configuration
    BATCH_SIZE = 32
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {DEVICE}")

    # Load preprocessed data
    X_train, y_train, X_val, y_val, X_test, y_test = load_preprocessed_data(
        data_dir='data/processed',
        device=DEVICE
    )

    # Create PyTorch preprocessor
    preprocessor = PyTorchPreprocessor()

    # Create DataLoaders
    loaders = preprocessor.create_dataloaders(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        batch_size=BATCH_SIZE,
        shuffle_train=True,
        device=DEVICE
    )

    # Demonstrate data loading
    print("\n" + "="*70)
    print("TESTING DATA LOADING")
    print("="*70)

    # Get one batch from train loader
    train_loader = loaders['train']
    X_batch, y_batch = next(iter(train_loader))

    print(f"\nBatch shapes:")
    print(f"  Features: {X_batch.shape}")
    print(f"  Targets:  {y_batch.shape}")
    print(f"  Device:   {X_batch.device}")

    # Show target distribution in batch
    print(f"\nBatch target distribution:")
    unique, counts = torch.unique(y_batch, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"  Class {label}: {count} samples ({count/len(y_batch)*100:.1f}%)")

    # Calculate class weights for loss function
    class_weights = get_class_weights(y_train, device=DEVICE)
    print(f"\nClass weights for weighted loss:")
    print(f"  Class 0 (no default): {class_weights[0]:.4f}")
    print(f"  Class 1 (default):    {class_weights[1]:.4f}")

    # Demonstrate iteration through full dataset
    print("\n" + "="*70)
    print("ITERATING THROUGH DATASET")
    print("="*70)

    total_samples = 0
    total_defaults = 0

    print("\nProcessing batches...")
    for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
        total_samples += len(y_batch)
        total_defaults += y_batch.sum().item()

        if batch_idx < 3:  # Show first 3 batches
            print(f"  Batch {batch_idx+1}: {X_batch.shape}, "
                  f"{y_batch.sum().item()} defaults")

    print(f"\nTotal samples processed: {total_samples}")
    print(f"Total defaults: {total_defaults} ({total_defaults/total_samples*100:.1f}%)")

    print("\n" + "="*70)
    print("PYTORCH DATA PIPELINE READY!")
    print("="*70)
    print("\nKey Benefits:")
    print("  ✓ Efficient batching with DataLoader")
    print("  ✓ GPU acceleration support")
    print("  ✓ Automatic shuffling and sampling")
    print("  ✓ Memory-efficient for large datasets")
    print("  ✓ Easy integration with PyTorch models")
    print("\nNext Steps:")
    print("  1. Use these DataLoaders with PyTorch models")
    print("  2. Implement training loops with batched data")
    print("  3. Add data augmentation if needed")
    print("  4. Scale to larger datasets with streaming")


if __name__ == "__main__":
    main()
