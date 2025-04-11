"""
Visualization utilities for model performance and data analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix(cm, class_names, title='Confusion Matrix', cmap='Blues'):
    """
    Plot a confusion matrix with class names.
    
    Args:
        cm: Confusion matrix array from sklearn
        class_names: List of class names in order
        title: Title for the plot
        cmap: Colormap for the heatmap
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, 
                xticklabels=class_names, 
                yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    

def plot_training_history(history, title='Training History'):
    """
    Plot training and validation metrics over epochs.
    
    Args:
        history: Dictionary with keys 'train_loss', 'val_loss', 'train_acc', 'val_acc'
                Each contains a list of values per epoch
        title: Title for the plot
    """
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()


def plot_class_distribution(class_counts, class_names, title='Class Distribution'):
    """
    Plot the distribution of classes in the dataset.
    
    Args:
        class_counts: Dictionary with class indices as keys and counts as values
        class_names: Dictionary mapping class indices to names
        title: Title for the plot
    """
    # Convert to proper format for plotting
    indices = sorted(class_counts.keys())
    counts = [class_counts[idx] for idx in indices]
    names = [class_names.get(idx, f"Class {idx}") for idx in indices]
    
    plt.figure(figsize=(12, 6))
    plt.bar(names, counts)
    plt.title(title)
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('class_distribution.png')
    plt.close()