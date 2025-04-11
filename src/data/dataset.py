"""
Dataset-related functionality for loading and processing MFCC features.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import WeightedRandomSampler


class MFCCDataset(Dataset):
    """Custom Dataset for loading MFCC files and multi-class labels."""
    def __init__(self, mfcc_dir, filenames, labels, fixed_length):
        self.mfcc_dir = mfcc_dir
        self.filenames = filenames
        self.labels = labels
        self.fixed_length = fixed_length
        self.num_mfcc_coeffs = None  # Determined by the first loaded file

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        mfcc_path = os.path.join(self.mfcc_dir, self.filenames[idx])
        try:
            mfcc = np.load(mfcc_path)  # Shape: (num_coeffs, num_steps)
        except Exception as e:
            print(f"Error loading MFCC file {mfcc_path}: {e}. Returning zeros.")
            # Return dummy data matching expected dims
            if self.num_mfcc_coeffs is None:
                self.num_mfcc_coeffs = 13  # Guess common value
            mfcc = np.zeros((self.num_mfcc_coeffs, self.fixed_length))

        if self.num_mfcc_coeffs is None:
            self.num_mfcc_coeffs = mfcc.shape[0]

        # Ensure consistent number of coefficients
        if mfcc.shape[0] != self.num_mfcc_coeffs:
            print(f"Warning: File {self.filenames[idx]} has {mfcc.shape[0]} coeffs, expected {self.num_mfcc_coeffs}. Using zeros.")
            mfcc = np.zeros((self.num_mfcc_coeffs, mfcc.shape[1]))

        # --- Padding / Truncation ---
        current_length = mfcc.shape[1]
        if current_length > self.fixed_length:
            mfcc = mfcc[:, :self.fixed_length]  # Truncate
        elif current_length < self.fixed_length:
            padding_width = self.fixed_length - current_length
            mfcc = np.pad(mfcc, ((0, 0), (0, padding_width)), mode='constant')  # Pad time axis

        # --- Format for PyTorch CNN (Batch, Channel, Height, Width) ---
        mfcc_tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)  # Add channel dim

        # --- Label Format ---
        # Use long for CrossEntropyLoss
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)

        return mfcc_tensor, label_tensor


def create_data_loaders(train_dataset, val_dataset, test_dataset, batch_size, sample_weights=None):
    train_loader = None
    val_loader = None
    
    # Only create a train loader if train_dataset is provided
    if train_dataset is not None:
        if sample_weights:
            sampler = WeightedRandomSampler(
                sample_weights, 
                num_samples=len(sample_weights), 
                replacement=True
            )
            train_loader = DataLoader(
                train_dataset, 
                batch_size=batch_size, 
                sampler=sampler, 
                num_workers=2, 
                pin_memory=True
            )
        else:
            train_loader = DataLoader(
                train_dataset, 
                batch_size=batch_size, 
                shuffle=True, 
                num_workers=2, 
                pin_memory=True
            )
    
    # Only create a validation loader if val_dataset is provided
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=2, 
            pin_memory=True
        )
    
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=2, 
            pin_memory=True
        )
    
    return train_loader, val_loader, test_loader
