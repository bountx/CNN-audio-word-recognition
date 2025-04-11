"""
Dataset-related functionality for loading and processing MFCC features.
"""

import os
import numpy as np
import torch
import torchaudio.transforms as T # <-- Import torchaudio transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import WeightedRandomSampler


class MFCCDataset(Dataset):
    """Custom Dataset for loading MFCC files and multi-class labels.
    Includes optional SpecAugment for training.
    """
    def __init__(self, mfcc_dir, filenames, labels, fixed_length, apply_augmentation=False): # <-- Added apply_augmentation flag
        self.mfcc_dir = mfcc_dir
        self.filenames = filenames
        self.labels = labels
        self.fixed_length = fixed_length
        self.num_mfcc_coeffs = None  # Determined by the first loaded file
        self.apply_augmentation = apply_augmentation # <-- Store the flag

        # --- Initialize Augmentation Transforms (only if needed) ---
        if self.apply_augmentation:
            # --- Parameters for SpecAugment (These are hyperparameters you can tune!) ---
            # Based on num_mfcc_coeffs=39 reported in your logs
            freq_mask_param = 7 # Max number of frequency bands to mask (e.g., ~1/3 of coeffs)
            # Based on typical segment lengths, maybe mask up to 20-50 time steps?
            time_mask_param = 20 # Max number of time steps to mask

            self.freq_mask = T.FrequencyMasking(freq_mask_param=freq_mask_param)
            self.time_mask = T.TimeMasking(time_mask_param=time_mask_param)
            print(f"INFO: SpecAugment enabled for this dataset (FreqMask: {freq_mask_param}, TimeMask: {time_mask_param})")
        else:
            self.freq_mask = None
            self.time_mask = None

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        mfcc_path = os.path.join(self.mfcc_dir, self.filenames[idx])
        try:
            # Load as numpy array first
            mfcc_np = np.load(mfcc_path)  # Shape: (num_coeffs, num_steps)
        except FileNotFoundError:
             print(f"Error: MFCC file not found {mfcc_path}. Returning zeros.")
             # Ensure num_mfcc_coeffs is set if this is the first file attempt
             if self.num_mfcc_coeffs is None: self.num_mfcc_coeffs = 39 # Use the known value
             mfcc_np = np.zeros((self.num_mfcc_coeffs, 1)) # Create minimal numpy array
        except Exception as e:
            print(f"Error loading MFCC file {mfcc_path}: {e}. Returning zeros.")
            if self.num_mfcc_coeffs is None: self.num_mfcc_coeffs = 39 # Use the known value
            mfcc_np = np.zeros((self.num_mfcc_coeffs, 1))

        if self.num_mfcc_coeffs is None:
            self.num_mfcc_coeffs = mfcc_np.shape[0]
            # Also update augmentation params if they depend on num_coeffs and were defined with a default
            # (Not strictly necessary with current static param definition, but good practice if params were relative)

        # Ensure consistent number of coefficients
        if mfcc_np.shape[0] != self.num_mfcc_coeffs:
            print(f"Warning: File {self.filenames[idx]} has {mfcc_np.shape[0]} coeffs, expected {self.num_mfcc_coeffs}. Using zeros.")
            mfcc_np = np.zeros((self.num_mfcc_coeffs, mfcc_np.shape[1]))

        # --- Convert to Tensor ---
        # We do this early to use torch padding and augmentation transforms
        mfcc_tensor = torch.tensor(mfcc_np, dtype=torch.float32) # Shape: (num_coeffs, current_length)

        # --- Padding / Truncation (using torch functions) ---
        current_length = mfcc_tensor.shape[1]
        if current_length > self.fixed_length:
            mfcc_tensor = mfcc_tensor[:, :self.fixed_length]  # Truncate
        elif current_length < self.fixed_length:
            padding_width = self.fixed_length - current_length
            # Pad needs (padding_left, padding_right) for the last dimension
            mfcc_tensor = torch.nn.functional.pad(mfcc_tensor, (0, padding_width), mode='constant', value=0)

        # --- Apply SpecAugment (if enabled) ---
        if self.apply_augmentation and self.freq_mask and self.time_mask:
            mfcc_tensor = self.freq_mask(mfcc_tensor)
            mfcc_tensor = self.time_mask(mfcc_tensor)

        # --- Format for PyTorch CNN (Batch, Channel, Height, Width) ---
        mfcc_tensor = mfcc_tensor.unsqueeze(0)  # Add channel dim -> (1, num_coeffs, fixed_length)

        # --- Label Format ---
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)

        return mfcc_tensor, label_tensor


def create_data_loaders(train_dataset, val_dataset, test_dataset, batch_size, sample_weights=None):
    train_loader = None
    val_loader = None
    
    # Only create a train loader if train_dataset is provided
    if train_dataset is not None:
        train_loader = DataLoader(
                train_dataset, 
                batch_size=batch_size, 
                shuffle=True, 
                num_workers=4, # Increased workers slightly
                pin_memory=True,
                persistent_workers=True
            )
    # Only create a validation loader if val_dataset is provided
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=4, # Increased workers slightly
            pin_memory=True,
            persistent_workers=True
        )
    
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=4, # Increased workers slightly
            pin_memory=True,
            persistent_workers=True
        )
    
    return train_loader, val_loader, test_loader