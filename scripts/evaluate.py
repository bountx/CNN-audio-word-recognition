#!/usr/bin/env python
"""
Evaluation script for the trained speech recognition model.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

# Add parent directory to path to import project modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config
from src.data.preprocessing import (
    load_target_words, load_data_and_labels, split_data
)
from src.data.dataset import MFCCDataset, create_data_loaders
from src.models.cnn import SimpleCNN
from src.utils.training import evaluate_epoch
from src.utils.metrics import calculate_performance_metrics
from src.visualization.plots import plot_confusion_matrix


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
    
    # Split data to get test set
    try:
        (train_files, train_lbls), (val_files, val_lbls), (test_files, test_lbls) = split_data(
            all_filenames, all_labels, config.TRAIN_SIZE, config.VAL_SIZE, num_classes, idx_to_word
        )
    except ValueError as e:
        print(f"Error during data splitting: {e}")
        return
    
    if not test_files:
        print("No test files available. Using validation files for evaluation.")
        test_files, test_lbls = val_files, val_lbls
    
    # Determine number of MFCC coefficients
    if test_files:
        temp_ds = MFCCDataset(config.MFCC_DIR, [test_files[0]], [test_lbls[0]], config.FIXED_MFCC_LENGTH)
        try:
            _, _ = temp_ds[0]  # Trigger loading to determine num_mfcc_coeffs
            num_mfcc_coeffs = temp_ds.num_mfcc_coeffs
            if num_mfcc_coeffs is None:
                raise ValueError("Could not determine number of MFCC coefficients.")
            print(f"Determined number of MFCC coefficients: {num_mfcc_coeffs}")
        except Exception as e:
            print(f"Error determining MFCC coefficients: {e}")
            print("Attempting to guess MFCC coefficients = 13")
            num_mfcc_coeffs = 13
    else:
        print("Error: No files available for evaluation.")
        return
    
    # Create test dataset and dataloader
    test_dataset = MFCCDataset(config.MFCC_DIR, test_files, test_lbls, config.FIXED_MFCC_LENGTH)
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=2, 
        pin_memory=True
    ) # Get just the test loader
    
    # Initialize model
    model = SimpleCNN(
        num_mfcc_coeffs=num_mfcc_coeffs, 
        fixed_length=config.FIXED_MFCC_LENGTH, 
        num_classes=num_classes
    ).to(config.DEVICE)
    
    # Load best model
    if os.path.exists(config.BEST_MODEL_PATH):
        try:
            model.load_state_dict(torch.load(config.BEST_MODEL_PATH))
            print(f"Loaded model from {config.BEST_MODEL_PATH}")
        except Exception as e:
            print(f"Error loading model: {e}")
            return
    else:
        print(f"Model file not found: {config.BEST_MODEL_PATH}")
        return
    
    # Set loss function (for evaluation metrics)
    criterion = nn.CrossEntropyLoss()
    
    # Evaluate model
    print("\nEvaluating model on test set...")
    test_loss, test_acc, test_preds, test_labels = evaluate_epoch(model, test_loader, criterion, config.DEVICE)
    
    print(f"\nTest Results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Calculate detailed metrics
    try:
        metrics = calculate_performance_metrics(test_labels, test_preds, idx_to_word)
        
        print("\nClassification Report:")
        print(metrics["report"])
        
        print("\nConfusion Matrix:")
        print(metrics["confusion_matrix"])
        
        # Plot confusion matrix
        class_names = [idx_to_word.get(i, f"Class {i}") for i in range(num_classes)]
        plot_confusion_matrix(metrics["confusion_matrix"], class_names)
        print("Confusion matrix saved to 'confusion_matrix.png'")
        
    except Exception as e:
        print(f"Error calculating performance metrics: {e}")


if __name__ == "__main__":
    main()