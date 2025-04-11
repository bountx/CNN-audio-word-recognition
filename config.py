"""
Configuration settings for the speech recognition project.
"""

import os
import torch

# File paths
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
TSV_FILE_PATH = os.path.join(DATA_DIR, 'labels.tsv')
MFCC_DIR = os.path.join(DATA_DIR, 'output_mfcc')
MFCC_EXTENSION = '.npy'
TARGET_WORDS_FILE = os.path.join(DATA_DIR, 'target_words.txt')
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), 'checkpoints')
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, 'best_model.bin')

# MFCC parameters
FIXED_MFCC_LENGTH = 47  # Adjust based on your MFCC extraction

# Data split parameters
TRAIN_SIZE = 40000
VAL_SIZE = 4000
# Test size will be the remainder

# Training parameters
BATCH_SIZE = 32
LEARNING_RATE = 0.0005
NUM_EPOCHS = 25
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MFCC_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)