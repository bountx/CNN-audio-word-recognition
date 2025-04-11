"""
Functions for data preprocessing, including loading TSV, creating labels,
and splitting data into train/val/test sets.
"""

import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_target_words(filepath):
    """Loads target words from a file, one word per line."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Target words file not found: {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            target_words = [line.strip().lower() for line in f if line.strip()]
        if not target_words:
            raise ValueError(f"No target words found in {filepath}")

        # Create mappings
        # Class 0: background (no target word found)
        # Class 1: target_words[0]
        # Class 2: target_words[1]
        # ...
        word_to_idx = {word: i + 1 for i, word in enumerate(target_words)}
        idx_to_word = {i + 1: word for i, word in enumerate(target_words)}
        idx_to_word[0] = 'background'  # Add background class representation
        num_classes = len(target_words) + 1

        print(f"Loaded {len(target_words)} target words: {target_words}")
        print(f"Number of classes: {num_classes} (including background)")
        print(f"Word to Index mapping: {word_to_idx}")

        # Create padded versions for searching
        target_words_padded = {f" {word} ": idx for word, idx in word_to_idx.items()}

        return target_words, word_to_idx, idx_to_word, target_words_padded, num_classes

    except Exception as e:
        print(f"Error loading target words from {filepath}: {e}")
        raise


def load_data_and_labels(tsv_path, mfcc_dir, mfcc_ext, target_words_padded, idx_to_word, background_idx=0):
    """Loads TSV, extracts filenames and creates multi-class labels."""
    try:
        # Assuming tab-separated with no header row
        df = pd.read_csv(tsv_path, sep='\t', header=None, on_bad_lines='skip', encoding='utf-8')
        df.columns = ['filename_wav', 'start', 'end', 'text']
    except Exception as e:
        print(f"Error reading TSV file {tsv_path}: {e}")
        return [], []

    filenames = []
    labels = []
    num_classes = len(idx_to_word)
    class_counts = {i: 0 for i in range(num_classes)}  # Initialize counts for all classes
    print(f"Processing {len(df)} rows from {tsv_path}...")

    for index, row in df.iterrows():
        wav_filename = row['filename_wav']
        text = f" {str(row['text']).lower()} "  # Pad text for boundary checks

        # Derive MFCC filename
        base_name = os.path.splitext(wav_filename)[0]
        mfcc_filename = base_name + mfcc_ext
        mfcc_full_path = os.path.join(mfcc_dir, mfcc_filename)

        # Check if the corresponding MFCC file actually exists
        if os.path.exists(mfcc_full_path):
            found_word_idx = background_idx  # Default to background
            # Check for each target word (first match wins for multi-class)
            for padded_word, idx in target_words_padded.items():
                if padded_word in text:
                    found_word_idx = idx
                    break  # Assign the index of the first target word found

            filenames.append(mfcc_filename)
            labels.append(found_word_idx)
            class_counts[found_word_idx] += 1

    print(f"Found {len(filenames)} segments with existing MFCC files.")
    print("Class distribution:")
    for idx, count in class_counts.items():
        word = idx_to_word.get(idx, f"Unknown Index {idx}")
        print(f"  Class {idx} ({word}): {count} samples")

    if not filenames:
        raise ValueError("No valid MFCC files found based on the TSV and MFCC directory.")
    return filenames, labels


def split_data(filenames, labels, train_size, val_size, num_classes, idx_to_word):
    """Splits filenames and labels into train, val, test sets, attempting stratification."""
    if len(filenames) < train_size + val_size:
        raise ValueError(f"Not enough data ({len(filenames)}) for the requested train ({train_size}) and val ({val_size}) sizes.")

    # Combine filenames and labels for consistent shuffling
    combined = list(zip(filenames, labels))
    random.shuffle(combined)
    shuffled_filenames, shuffled_labels = zip(*combined)
    shuffled_labels = list(shuffled_labels)  # Convert back to list for splitting

    # Check if stratification is feasible
    unique_labels, counts = np.unique(shuffled_labels, return_counts=True)
    min_samples_per_class = counts.min()
    can_stratify = min_samples_per_class >= 2  # Need at least 2 samples per class for stratified split

    stratify_option = shuffled_labels if can_stratify else None
    if not can_stratify:
        print("Warning: Cannot stratify split due to classes with < 2 samples. Performing random split.")

    # First split: Train + Temp (Val + Test)
    try:
        train_filenames, temp_filenames, train_labels, temp_labels = train_test_split(
            shuffled_filenames, shuffled_labels,
            train_size=train_size,
            random_state=42,
            stratify=stratify_option
        )
    except ValueError as e:
        print(f"Warning during first split (possibly stratification issue): {e}. Trying without stratification.")
        train_filenames, temp_filenames, train_labels, temp_labels = train_test_split(
            shuffled_filenames, shuffled_labels,
            train_size=train_size,
            random_state=42,
            stratify=None
        )

    # Calculate remaining size and required val size relative to temp set
    remaining_size = len(temp_filenames)
    if remaining_size < val_size:
        # Option 1: Reduce validation size
        print(f"Warning: Not enough remaining data ({remaining_size}) for requested validation size ({val_size}). Reducing validation size to {remaining_size}.")
        actual_val_size = remaining_size
    else:
        actual_val_size = val_size

    test_size_fraction = (remaining_size - actual_val_size) / remaining_size if remaining_size > 0 and remaining_size > actual_val_size else 0
    val_size_fraction = actual_val_size / remaining_size if remaining_size > 0 else 1  # Should not happen due to check above

    # Check stratification feasibility for the second split
    unique_labels_temp, counts_temp = np.unique(temp_labels, return_counts=True)
    min_samples_per_class_temp = counts_temp.min() if len(counts_temp) > 0 else 0
    can_stratify_temp = min_samples_per_class_temp >= 2 and test_size_fraction > 0 and test_size_fraction < 1
    stratify_option_temp = temp_labels if can_stratify_temp else None
    if not can_stratify_temp and test_size_fraction > 0 and test_size_fraction < 1:
        print("Warning: Cannot stratify second split (val/test). Performing random split.")

    if remaining_size > 0 and actual_val_size < remaining_size:
        # Second split: Val + Test from Temp
        try:
            val_filenames, test_filenames, val_labels, test_labels = train_test_split(
                temp_filenames, temp_labels,
                test_size=test_size_fraction,  # Fraction to become test set
                random_state=42,
                stratify=stratify_option_temp
            )
        except ValueError as e:
            print(f"Warning during second split (possibly stratification issue): {e}. Trying without stratification.")
            val_filenames, test_filenames, val_labels, test_labels = train_test_split(
                temp_filenames, temp_labels,
                test_size=test_size_fraction,
                random_state=42,
                stratify=None
            )

    elif actual_val_size == remaining_size and remaining_size > 0:
        # Use all remaining as validation if test size becomes 0
        val_filenames, test_filenames, val_labels, test_labels = temp_filenames, [], temp_labels, []
        print("Warning: Test set is empty as all remaining data used for validation.")
    else:  # remaining_size is 0
        val_filenames, test_filenames, val_labels, test_labels = [], [], [], []
        print("Warning: Both validation and test sets are empty.")

    print(f"Train set size: {len(train_filenames)}")
    print(f"Validation set size: {len(val_filenames)}")
    print(f"Test set size: {len(test_filenames)}")

    # Print class distribution for each set
    for name, labels_set in [("Train", train_labels), ("Validation", val_labels), ("Test", test_labels)]:
        if labels_set:
            unique, counts = np.unique(labels_set, return_counts=True)
            dist_str = ", ".join([f"{idx_to_word.get(l, '?')}({l}):{c}" for l, c in zip(unique, counts)])
            print(f"  {name} distribution: {dist_str}")
        else:
            print(f"  {name} distribution: Empty")

    return (train_filenames, train_labels), \
           (val_filenames, val_labels), \
           (test_filenames, test_labels)


def calculate_class_weights(labels, num_classes):
    """Calculate class weights inversely proportional to class frequency."""
    class_counts = np.bincount(labels, minlength=num_classes)
    print(f"Class counts: {class_counts}")
    
    # Compute weights: inverse frequency (add a small constant to avoid division by zero)
    weights = 1.0 / (class_counts + 1e-6)
    
    # Print computed weights for debugging
    print(f"Per-class weights: {weights}")
    
    return weights