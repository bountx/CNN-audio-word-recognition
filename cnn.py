import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import random
from sklearn.model_selection import train_test_split # For easier splitting

# --- Configuration ---
TSV_FILE_PATH = 'labels.tsv'
MFCC_DIR = 'output_mfcc'
MFCC_EXTENSION = '.npy'
TARGET_WORDS_FILE = 'target_words.txt' # File containing target words, one per line

FIXED_MFCC_LENGTH = 47 # Adjust based on your MFCC extraction

# Data Splits
TRAIN_SIZE = 40000
VAL_SIZE = 4000
# Test size will be the remainder

BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- 0. Load Target Words and Create Mappings ---
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
        idx_to_word[0] = 'background' # Add background class representation
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

# Load target words globally or pass them around
try:
    TARGET_WORDS, WORD_TO_IDX, IDX_TO_WORD, TARGET_WORDS_PADDED, NUM_CLASSES = load_target_words(TARGET_WORDS_FILE)
except Exception as e:
    print(f"Exiting due to error loading target words: {e}")
    exit()

# --- 1. Load Data and Create Labels ---
def load_data_and_labels(tsv_path, mfcc_dir, mfcc_ext, target_words_padded, background_idx=0):
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
    class_counts = {i: 0 for i in range(NUM_CLASSES)} # Initialize counts for all classes
    print(f"Processing {len(df)} rows from {tsv_path}...")

    for index, row in df.iterrows():
        wav_filename = row['filename_wav']
        text = f" {str(row['text']).lower()} " # Pad text for boundary checks

        # Derive MFCC filename
        base_name = os.path.splitext(wav_filename)[0]
        mfcc_filename = base_name + mfcc_ext
        mfcc_full_path = os.path.join(mfcc_dir, mfcc_filename)

        # Check if the corresponding MFCC file actually exists
        if os.path.exists(mfcc_full_path):
            found_word_idx = background_idx # Default to background
            # Check for each target word (first match wins for multi-class)
            for padded_word, idx in target_words_padded.items():
                if padded_word in text:
                    found_word_idx = idx
                    break # Assign the index of the first target word found

            filenames.append(mfcc_filename)
            labels.append(found_word_idx)
            class_counts[found_word_idx] += 1
        else:
            # Optional: Reduce verbosity if many files are missing
            # if index % 1000 == 0:
            #     print(f"Warning: MFCC file not found for {wav_filename}, skipping (logged once per 1000).")
            pass # Silently skip if MFCC not found


    print(f"Found {len(filenames)} segments with existing MFCC files.")
    print("Class distribution:")
    for idx, count in class_counts.items():
         word = IDX_TO_WORD.get(idx, f"Unknown Index {idx}")
         print(f"  Class {idx} ({word}): {count} samples")

    if not filenames:
         raise ValueError("No valid MFCC files found based on the TSV and MFCC directory.")
    return filenames, labels

# --- 2. Split Data ---
def split_data(filenames, labels, train_size, val_size, num_classes):
    """Splits filenames and labels into train, val, test sets, attempting stratification."""
    if len(filenames) < train_size + val_size:
        raise ValueError(f"Not enough data ({len(filenames)}) for the requested train ({train_size}) and val ({val_size}) sizes.")

    # Combine filenames and labels for consistent shuffling
    combined = list(zip(filenames, labels))
    random.shuffle(combined)
    shuffled_filenames, shuffled_labels = zip(*combined)
    shuffled_labels = list(shuffled_labels) # Convert back to list for splitting

    # Check if stratification is feasible
    unique_labels, counts = np.unique(shuffled_labels, return_counts=True)
    min_samples_per_class = counts.min()
    can_stratify = min_samples_per_class >= 2 # Need at least 2 samples per class for stratified split

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
         # Option 2: Raise error (uncomment if preferred)
         # raise ValueError(f"Not enough remaining data ({remaining_size}) for validation set ({val_size}).")

    else:
        actual_val_size = val_size

    test_size_fraction = (remaining_size - actual_val_size) / remaining_size if remaining_size > 0 and remaining_size > actual_val_size else 0
    val_size_fraction = actual_val_size / remaining_size if remaining_size > 0 else 1 # Should not happen due to check above

    # Check stratification feasibility for the second split
    unique_labels_temp, counts_temp = np.unique(temp_labels, return_counts=True)
    min_samples_per_class_temp = counts_temp.min() if len(counts_temp)>0 else 0
    can_stratify_temp = min_samples_per_class_temp >= 2 and test_size_fraction > 0 and test_size_fraction < 1
    stratify_option_temp = temp_labels if can_stratify_temp else None
    if not can_stratify_temp and test_size_fraction > 0 and test_size_fraction < 1:
         print("Warning: Cannot stratify second split (val/test). Performing random split.")


    if remaining_size > 0 and actual_val_size < remaining_size :
         # Second split: Val + Test from Temp
         try:
            val_filenames, test_filenames, val_labels, test_labels = train_test_split(
                temp_filenames, temp_labels,
                test_size=test_size_fraction, # Fraction to become test set
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
    else: # remaining_size is 0
        val_filenames, test_filenames, val_labels, test_labels = [], [], [], []
        print("Warning: Both validation and test sets are empty.")


    print(f"Train set size: {len(train_filenames)}")
    print(f"Validation set size: {len(val_filenames)}")
    print(f"Test set size: {len(test_filenames)}")

    # Print class distribution for each set
    for name, labels_set in [("Train", train_labels), ("Validation", val_labels), ("Test", test_labels)]:
        if labels_set:
            unique, counts = np.unique(labels_set, return_counts=True)
            dist_str = ", ".join([f"{IDX_TO_WORD.get(l, '?')}({l}):{c}" for l, c in zip(unique, counts)])
            print(f"  {name} distribution: {dist_str}")
        else:
            print(f"  {name} distribution: Empty")


    return (train_filenames, train_labels), \
           (val_filenames, val_labels), \
           (test_filenames, test_labels)


# --- 3. PyTorch Dataset ---
class MFCCDataset(Dataset):
    """Custom Dataset for loading MFCC files and multi-class labels."""
    def __init__(self, mfcc_dir, filenames, labels, fixed_length):
        self.mfcc_dir = mfcc_dir
        self.filenames = filenames
        self.labels = labels
        self.fixed_length = fixed_length
        self.num_mfcc_coeffs = None # Determined by the first loaded file

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        mfcc_path = os.path.join(self.mfcc_dir, self.filenames[idx])
        try:
            mfcc = np.load(mfcc_path) # Shape: (num_coeffs, num_steps)
        except Exception as e:
            print(f"Error loading MFCC file {mfcc_path}: {e}. Returning zeros.")
            # Return dummy data matching expected dims
            if self.num_mfcc_coeffs is None: self.num_mfcc_coeffs = 13 # Guess common value
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
            mfcc = mfcc[:, :self.fixed_length] # Truncate
        elif current_length < self.fixed_length:
            padding_width = self.fixed_length - current_length
            mfcc = np.pad(mfcc, ((0, 0), (0, padding_width)), mode='constant') # Pad time axis

        # --- Format for PyTorch CNN (Batch, Channel, Height, Width) ---
        mfcc_tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0) # Add channel dim

        # --- Label Format ---
        # Use long for CrossEntropyLoss
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)

        return mfcc_tensor, label_tensor

# --- 4. CNN Model Definition ---
class SimpleCNN(nn.Module):
    def __init__(self, num_mfcc_coeffs, fixed_length, num_classes): # Add num_classes
        super(SimpleCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            # Input shape: (batch, 1, num_coeffs, fixed_length)
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 5), padding=(1, 2)),
            nn.BatchNorm2d(16), # Add Batch Norm
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 5), padding=(1, 2)),
            nn.BatchNorm2d(32), # Add Batch Norm
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64), # Add Batch Norm
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )

        # Calculate the flattened size dynamically
        dummy_input = torch.randn(1, 1, num_mfcc_coeffs, fixed_length)
        dummy_output = self.conv_layers(dummy_input)
        self.flattened_size = int(np.prod(dummy_output.shape))

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattened_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes) # Output num_classes logits
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x # Return raw logits

# --- 5. Training and Evaluation Functions ---
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device) # Labels are now Long, no need to unsqueeze for CrossEntropyLoss

        optimizer.zero_grad()
        outputs = model(inputs) # Raw logits
        loss = criterion(outputs, labels) # CrossEntropyLoss expects logits and class indices
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1) # Get the index of the max logit == predicted class
        correct_predictions += torch.sum(preds == labels).item()
        total_samples += labels.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples
    return epoch_loss, epoch_acc

def evaluate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == labels).item()
            total_samples += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())


    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples
    return epoch_loss, epoch_acc, all_preds, all_labels # Return predictions for analysis

# --- Main Execution ---
if __name__ == "__main__":
    # 1. Load and Label
    all_filenames, all_labels = load_data_and_labels(
        TSV_FILE_PATH, MFCC_DIR, MFCC_EXTENSION, TARGET_WORDS_PADDED
    )

    if not all_filenames:
        print("Exiting: No data loaded.")
        exit()

    # 2. Split Data
    try:
        (train_files, train_lbls), (val_files, val_lbls), (test_files, test_lbls) = split_data(
            all_filenames, all_labels, TRAIN_SIZE, VAL_SIZE, NUM_CLASSES
        )
    except ValueError as e:
        print(f"Error during data splitting: {e}")
        exit()

    # --- Calculate Class Weights for Training Set ---
    print("\nCalculating class weights for training set...")
    train_class_counts = np.bincount(train_lbls, minlength=NUM_CLASSES)
    total_train_samples = len(train_lbls)

    # Handle potential zero counts to avoid division by zero if a class is missing in train
    # (Though split_data tries to prevent this with stratification)
    class_weights = []
    for count in train_class_counts:
        if count == 0:
            # Assign weight of 1 or 0 if class truly absent? Let's use 1 for now.
            # Or handle more gracefully if this scenario is expected.
             print(f"Warning: Class with 0 samples found in training set. Assigning weight 1.0")
             class_weights.append(1.0)
        else:
            weight = total_train_samples / (NUM_CLASSES * count)
            class_weights.append(weight)

    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
    print(f"Calculated class weights: {class_weights_tensor.numpy()}")
    # --- End Class Weight Calculation ---


    # 3. Create Datasets and DataLoaders
    # (Rest of the dataset/dataloader creation code remains the same...)
    # Determine num_mfcc_coeffs...
    if train_files:
        temp_ds = MFCCDataset(MFCC_DIR, [train_files[0]], [train_lbls[0]], FIXED_MFCC_LENGTH)
        try:
            _, _ = temp_ds[0] # Trigger loading and setting num_mfcc_coeffs
            NUM_MFCC_COEFFS = temp_ds.num_mfcc_coeffs
            if NUM_MFCC_COEFFS is None:
                raise ValueError("Could not determine number of MFCC coefficients.")
            print(f"Determined number of MFCC coefficients: {NUM_MFCC_COEFFS}")
        except Exception as e:
            print(f"Error determining MFCC coefficients from first file '{train_files[0]}': {e}")
            print("Attempting to guess MFCC coefficients = 13")
            NUM_MFCC_COEFFS = 13
    elif all_filenames:
         # If no training files but other files exist (edge case)
         temp_ds = MFCCDataset(MFCC_DIR, [all_filenames[0]], [all_labels[0]], FIXED_MFCC_LENGTH)
         try:
             _, _ = temp_ds[0]
             NUM_MFCC_COEFFS = temp_ds.num_mfcc_coeffs
             if NUM_MFCC_COEFFS is None: raise ValueError("Could not determine number of MFCC coefficients.")
             print(f"Determined number of MFCC coefficients (from first available file): {NUM_MFCC_COEFFS}")
         except Exception as e:
             print(f"Error determining MFCC coefficients: {e}")
             print("Attempting to guess MFCC coefficients = 13")
             NUM_MFCC_COEFFS = 13
    else:
        print("Error: No files available to determine MFCC coefficients.")
        exit()


    train_dataset = MFCCDataset(MFCC_DIR, train_files, train_lbls, FIXED_MFCC_LENGTH)
    val_dataset = MFCCDataset(MFCC_DIR, val_files, val_lbls, FIXED_MFCC_LENGTH)
    test_dataset = MFCCDataset(MFCC_DIR, test_files, test_lbls, FIXED_MFCC_LENGTH) if test_files else None

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True) if train_dataset else None
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True) if val_dataset else None
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True) if test_dataset else None

    if not train_loader or not val_loader:
        print("Error: Cannot proceed without training and validation data.")
        exit()

    # 4. Initialize Model, Loss, Optimizer
    model = SimpleCNN(num_mfcc_coeffs=NUM_MFCC_COEFFS, fixed_length=FIXED_MFCC_LENGTH, num_classes=NUM_CLASSES).to(DEVICE)
    print(model)

    # Use CrossEntropyLoss with calculated weights
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor.to(DEVICE)) # Pass weights here!
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- 6. Training Loop ---
    # (Training loop remains the same)
    best_val_loss = float('inf')
    print("\nStarting Training (with weighted loss)...")
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        # Note: Accuracy might appear lower during training now because the loss
        # prioritizes minority classes, potentially making more mistakes on the majority class.
        # Focus on validation loss and the final per-class metrics.
        val_loss, val_acc, _, _ = evaluate_epoch(model, val_loader, criterion, DEVICE)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_multiword_model_state_weighted.bin') # Use a new name
            print(f"Validation loss improved. Saved model checkpoint.")

    print("Training finished.")


    # --- 7. Final Evaluation on Test Set ---
    model_path = 'best_multiword_model_state_weighted.bin'
    if test_loader:
         print(f"\nLoading best model ({model_path}) for testing...")
         if os.path.exists(model_path):
             try:
                 model.load_state_dict(torch.load(model_path))
                 test_loss, test_acc, test_preds, test_labels = evaluate_epoch(model, test_loader, criterion, DEVICE)
                 print(f"\nTest Results (Weighted Loss Model):")
                 print(f"Test Loss: {test_loss:.4f}")
                 print(f"Test Accuracy: {test_acc:.4f}") # Accuracy will likely be lower than before!

                 # Optional: More detailed analysis (e.g., confusion matrix)
                 try:
                     from sklearn.metrics import confusion_matrix, classification_report
                     print("\nClassification Report:")
                     target_names = [IDX_TO_WORD.get(i, f"Class {i}") for i in range(NUM_CLASSES)]
                     valid_indices = set(range(NUM_CLASSES))
                     filtered_labels = [l for l in test_labels if l in valid_indices]
                     filtered_preds = [p for i, p in enumerate(test_preds) if test_labels[i] in valid_indices]

                     if filtered_labels and filtered_preds:
                         print(classification_report(filtered_labels, filtered_preds, target_names=target_names, zero_division=0))
                         print("\nConfusion Matrix:")
                         cm = confusion_matrix(filtered_labels, filtered_preds, labels=list(range(NUM_CLASSES)))
                         print(cm)
                     else:
                         print("Could not generate classification report/confusion matrix.")

                 except ImportError:
                     print("\nInstall scikit-learn (`pip install scikit-learn`) for classification report and confusion matrix.")
                 except Exception as report_e:
                      print(f"Error generating classification report/confusion matrix: {report_e}")

             except Exception as e:
                  print(f"An error occurred during testing: {e}")
         else:
              print(f"Could not find '{model_path}'. Cannot test.")
    else:
        print("\nSkipping testing: Test set is empty or test loader not created.")