#!/usr/bin/env python
"""
Training script for the speech recognition model.
"""

from collections import Counter
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Add parent directory to path to import project modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config
from src.data.preprocessing import (
    load_target_words, load_data_and_labels, split_data, calculate_class_weights
)
from src.data.dataset import MFCCDataset, create_data_loaders
from src.models.cnn import SimpleCNN
from src.utils.training import train_epoch, evaluate_epoch
from src.visualization.plots import plot_training_history
from src.utils.focal_loss import FocalLoss


def main():
    print(f"Using device: {config.DEVICE}")
    
    # Load target words and class mappings
    target_words, word_to_idx, idx_to_word, target_words_padded, num_classes = load_target_words(
        config.TARGET_WORDS_FILE
    )
    
    # Load data and create labels
    all_filenames, all_labels = load_data_and_labels(
        config.TSV_FILE_PATH, 
        config.MFCC_DIR, 
        config.MFCC_EXTENSION, 
        target_words_padded,
        idx_to_word
    )
    
    if not all_filenames:
        print("Exiting: No data loaded.")
        return

    # Split data into train/val/test sets
    try:
        (train_files, train_lbls), (val_files, val_lbls), (test_files, test_lbls) = split_data(
            all_filenames, all_labels, config.TRAIN_SIZE, config.VAL_SIZE, num_classes, idx_to_word
        )
    except ValueError as e:
        print(f"Error during data splitting: {e}")
        return
    
    class_counts = np.array([count for label, count in sorted(Counter(train_lbls).items())]) # Make sure class_counts is defined
    
    # Create datasets
    train_dataset = MFCCDataset(
        config.MFCC_DIR, train_files, train_lbls, config.FIXED_MFCC_LENGTH,
        apply_augmentation=True # <-- SET TO TRUE
    )
    val_dataset = MFCCDataset(
        config.MFCC_DIR, val_files, val_lbls, config.FIXED_MFCC_LENGTH,
        apply_augmentation=False # <-- SET TO FALSE
    )
    test_dataset = MFCCDataset(
        config.MFCC_DIR, test_files, test_lbls, config.FIXED_MFCC_LENGTH,
        apply_augmentation=False # <-- SET TO FALSE
    ) if test_files else None

        # Determine number of MFCC coefficients from first file
    if train_dataset and len(train_dataset) > 0:
        try:
            # Accessing __getitem__ might trigger num_mfcc_coeffs determination
            _, _ = train_dataset[0] 
            num_mfcc_coeffs = train_dataset.num_mfcc_coeffs
            if num_mfcc_coeffs is None:
                # Fallback if __getitem__ failed or didn't set it
                print("Could not determine num_mfcc_coeffs from dataset, guessing 39.")
                num_mfcc_coeffs = 39
            print(f"Determined number of MFCC coefficients: {num_mfcc_coeffs}")
        except Exception as e:
            print(f"Error determining MFCC coefficients from dataset: {e}")
            print("Attempting to guess MFCC coefficients = 39")
            num_mfcc_coeffs = 39 # Use the known value
    else:
        print("Error: Training dataset is empty or not created. Cannot determine MFCC coefficients.")
        # Handle error - maybe exit or use a default
        num_mfcc_coeffs = 39 # Use the known value as fallback
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, val_dataset, test_dataset, 
        config.BATCH_SIZE
    )
    if not train_loader or not val_loader:
        print("Error: Cannot proceed without training and validation data.")
        return
        
    # Initialize model, loss function and optimizer
    model = SimpleCNN(
        num_mfcc_coeffs=num_mfcc_coeffs, 
        fixed_length=config.FIXED_MFCC_LENGTH, 
        num_classes=num_classes
    ).to(config.DEVICE)
    
    print(model)
    
    gamma_value = 0.25 # Common starting point, tune if necessary (e.g., 1.0, 3.0)
    criterion = FocalLoss(alpha=None, gamma=gamma_value, reduction='mean', epsilon=1e-7)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
    
    # Training loop
    best_val_loss = float('inf')
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    print("\nStarting Training (with weighted loss)...")
    for epoch in range(config.NUM_EPOCHS):
        # Train for one epoch
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, config.DEVICE
        )
        
        # Evaluate on validation set
        val_loss, val_acc, _, _ = evaluate_epoch(
            model, val_loader, criterion, config.DEVICE
        )
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS} | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), config.BEST_MODEL_PATH)
            print(f"Validation loss improved. Saved model checkpoint.")
    
    print("Training finished.")
    
    # Plot training history
    try:
        from src.visualization.plots import plot_training_history
        plot_training_history(history, 'Model Training History')
        print("Training history plot saved to 'training_history.png'")
    except Exception as e:
        print(f"Could not create training history plot: {e}")


if __name__ == "__main__":
    main()