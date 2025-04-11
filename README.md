# Speech Recognition - Keyword Detection

This project uses a CNN-based approach to detect specific target words in audio segments, using MFCC features extracted from audio files.

## Project Overview

This system processes audio segments to detect the presence of specific target words. It uses:
- MFCC (Mel-frequency cepstral coefficients) as audio features
- A CNN (Convolutional Neural Network) architecture for classification
- Multi-class classification to identify specific words or "background" (no target word)

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Prepare your data:
   - Place your TSV file with format: `filename_wav, start, end, text` in `data/labels.tsv`
   - Create a file with target words (one per line) in `data/target_words.txt`
   - Place MFCC features as .npy files in `data/output_mfcc/`

## Usage

### Training

```bash
python scripts/train.py
```

### Evaluation

```bash
python scripts/evaluate.py
```

## Model Architecture

The model uses a 3-layer CNN followed by fully connected layers:
- 3 convolutional blocks (conv → batch norm → ReLU → max pool)
- 2 fully connected layers with dropout for regularization
- Output layer with one neuron per class (target words + background)

## Class Imbalance Handling

The system handles class imbalance using:
- Weighted loss function (inversely proportional to class frequency)
- Weighted random sampling during training