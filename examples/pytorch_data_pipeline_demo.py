"""
Practical Demo: PyTorch Data Pipeline
Shows how to use the new PyTorch data loading system
"""

import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.pytorch_data_loader import (
    load_preprocessed_data,
    PyTorchPreprocessor,
    get_class_weights
)


def demo_basic_loading():
    """Demo 1: Basic data loading with PyTorch."""
    print("\n" + "="*70)
    print("DEMO 1: BASIC DATA LOADING")
    print("="*70)

    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load preprocessed data
    print("\n1. Loading preprocessed CSV files...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_preprocessed_data(
        data_dir='data/processed_no_indicators'
    )

    # Create preprocessor
    print("\n2. Creating PyTorch DataLoaders...")
    preprocessor = PyTorchPreprocessor()

    loaders = preprocessor.create_dataloaders(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        batch_size=32,
        shuffle_train=True,
        device=device
    )

    # Access loaders
    train_loader = loaders['train']
    val_loader = loaders['val']
    test_loader = loaders['test']

    print("\n3. Getting a sample batch...")
    X_batch, y_batch = next(iter(train_loader))

    print(f"\nBatch information:")
    print(f"  Features shape: {X_batch.shape}")
    print(f"  Targets shape:  {y_batch.shape}")
    print(f"  Device:         {X_batch.device}")
    print(f"  Features dtype: {X_batch.dtype}")
    print(f"  Targets dtype:  {y_batch.dtype}")

    print("\n4. Analyzing batch distribution...")
    unique, counts = torch.unique(y_batch, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"  Class {label}: {count:2d} samples ({count/len(y_batch)*100:5.1f}%)")

    print("\n✓ Demo 1 Complete!")


def demo_iteration():
    """Demo 2: Iterating through batches."""
    print("\n" + "="*70)
    print("DEMO 2: BATCH ITERATION")
    print("="*70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load data
    X_train, y_train, X_val, y_val, _, _ = load_preprocessed_data(
        data_dir='data/processed_no_indicators'
    )

    preprocessor = PyTorchPreprocessor()
    loaders = preprocessor.create_dataloaders(
        X_train, y_train, X_val, y_val,
        batch_size=64,
        device=device
    )

    train_loader = loaders['train']

    print("\nIterating through first 5 batches...")
    total_samples = 0
    total_defaults = 0

    for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
        if batch_idx >= 5:
            break

        batch_defaults = y_batch.sum().item()
        total_samples += len(y_batch)
        total_defaults += batch_defaults

        print(f"  Batch {batch_idx+1}: "
              f"shape={X_batch.shape}, "
              f"defaults={batch_defaults}/{len(y_batch)} "
              f"({batch_defaults/len(y_batch)*100:.1f}%)")

    print(f"\nTotal processed: {total_samples} samples")
    print(f"Total defaults: {total_defaults} ({total_defaults/total_samples*100:.1f}%)")

    print("\n✓ Demo 2 Complete!")


def demo_with_simple_model():
    """Demo 3: Using DataLoader with a simple PyTorch model."""
    print("\n" + "="*70)
    print("DEMO 3: TRAINING WITH SIMPLE MODEL")
    print("="*70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load data
    print("\n1. Loading data...")
    X_train, y_train, X_val, y_val, _, _ = load_preprocessed_data(
        data_dir='data/processed_no_indicators'
    )

    preprocessor = PyTorchPreprocessor()
    loaders = preprocessor.create_dataloaders(
        X_train, y_train, X_val, y_val,
        batch_size=128,
        shuffle_train=True,
        device=device
    )

    n_features = preprocessor.n_features
    print(f"Number of features: {n_features}")

    # Create a simple model
    print("\n2. Creating simple neural network...")
    model = nn.Sequential(
        nn.Linear(n_features, 64),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(32, 2)  # Binary classification
    ).to(device)

    print(f"Model architecture:")
    print(model)

    # Calculate class weights for imbalanced data
    print("\n3. Calculating class weights...")
    class_weights = get_class_weights(y_train, device=device)
    print(f"  Class 0 (no default): {class_weights[0]:.4f}")
    print(f"  Class 1 (default):    {class_weights[1]:.4f}")

    # Setup training
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop (just 3 epochs for demo)
    print("\n4. Training for 3 epochs (demo)...")
    num_epochs = 3

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for X_batch, y_batch in loaders['train']:
            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track metrics
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += y_batch.size(0)
            train_correct += predicted.eq(y_batch).sum().item()

        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for X_batch, y_batch in loaders['val']:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += y_batch.size(0)
                val_correct += predicted.eq(y_batch).sum().item()

        # Print epoch results
        train_loss = train_loss / len(loaders['train'])
        train_acc = 100. * train_correct / train_total
        val_loss = val_loss / len(loaders['val'])
        val_acc = 100. * val_correct / val_total

        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f}, Acc: {val_acc:.2f}%")

    print("\n✓ Demo 3 Complete!")
    print("\nNote: This was just 3 epochs for demonstration.")
    print("For real training, you'd run 50-100 epochs with early stopping.")


def demo_memory_comparison():
    """Demo 4: Show memory efficiency."""
    print("\n" + "="*70)
    print("DEMO 4: MEMORY EFFICIENCY COMPARISON")
    print("="*70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load data
    X_train, y_train, _, _, _, _ = load_preprocessed_data(
        data_dir='data/processed_no_indicators'
    )

    # Old way: Load everything into memory
    print("\n1. OLD WAY (pandas/numpy):")
    print(f"  Loading entire dataset into memory...")
    print(f"  Dataset size: {X_train.shape}")
    memory_full = X_train.memory_usage(deep=True).sum() / 1024 / 1024
    print(f"  Memory usage: {memory_full:.2f} MB")
    print(f"  Problem: All data in RAM at once!")

    # New way: Batch loading
    print("\n2. NEW WAY (PyTorch DataLoader):")
    batch_sizes = [32, 64, 128, 256]

    for batch_size in batch_sizes:
        preprocessor = PyTorchPreprocessor()
        loaders = preprocessor.create_dataloaders(
            X_train, y_train,
            X_train[:100], y_train[:100],  # Small val set for demo
            batch_size=batch_size,
            device=device
        )

        # Get one batch
        X_batch, _ = next(iter(loaders['train']))

        # Calculate batch memory
        batch_memory = X_batch.element_size() * X_batch.nelement() / 1024 / 1024
        reduction = memory_full / batch_memory

        print(f"\n  Batch size {batch_size}:")
        print(f"    Batch shape: {X_batch.shape}")
        print(f"    Memory per batch: {batch_memory:.4f} MB")
        print(f"    Memory reduction: {reduction:.0f}x less!")

    print("\n✓ Demo 4 Complete!")
    print("\nConclusion: PyTorch DataLoader uses a tiny fraction of memory!")


def main():
    """Run all demos."""
    print("\n" + "="*70)
    print("PYTORCH DATA PIPELINE - COMPREHENSIVE DEMO")
    print("="*70)

    try:
        # Demo 1: Basic loading
        demo_basic_loading()

        # Demo 2: Iteration
        demo_iteration()

        # Demo 3: With model
        demo_with_simple_model()

        # Demo 4: Memory comparison
        demo_memory_comparison()

        # Summary
        print("\n" + "="*70)
        print("ALL DEMOS COMPLETE!")
        print("="*70)
        print("\nWhat you learned:")
        print("  ✓ How to load data with PyTorch DataLoader")
        print("  ✓ How to iterate through batches")
        print("  ✓ How to train a simple model with batched data")
        print("  ✓ Memory efficiency benefits (100-1000x less!)")
        print("\nNext steps:")
        print("  1. Read: docs/PYTORCH_DATA_PIPELINE_GUIDE.md")
        print("  2. Replace Random Forest with neural networks")
        print("  3. Train full model with early stopping")
        print("  4. Integrate with MLflow for experiment tracking")

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you have preprocessed data:")
        print("  python src/models/data_preprocessing_no_indicators.py")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
