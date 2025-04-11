"""
Metrics calculation and evaluation utilities.
"""

from sklearn.metrics import confusion_matrix, classification_report


def calculate_performance_metrics(true_labels, predictions, class_names=None):
    """
    Calculate and return detailed performance metrics.
    
    Args:
        true_labels: List of true class labels
        predictions: List of predicted class labels
        class_names: Dictionary mapping class indices to names
        
    Returns:
        Dictionary containing metrics and formatted report
    """
    # Filter out any invalid indices
    valid_indices = set(range(len(class_names))) if class_names else None
    
    if valid_indices:
        filtered_indices = [(i, l, p) for i, (l, p) in enumerate(zip(true_labels, predictions)) 
                           if l in valid_indices]
        
        if not filtered_indices:
            return {"error": "No valid labels found for metrics calculation"}
            
        indices, filtered_labels, filtered_preds = zip(*filtered_indices)
    else:
        filtered_labels, filtered_preds = true_labels, predictions
    
    # Create class names list if provided
    target_names = [class_names.get(i, f"Class {i}") for i in range(len(class_names))] if class_names else None
    
    # Generate classification report
    report = classification_report(
        filtered_labels, 
        filtered_preds,
        target_names=target_names,
        zero_division=0
    )
    
    # Generate confusion matrix
    cm = confusion_matrix(
        filtered_labels, 
        filtered_preds, 
        labels=list(range(len(class_names))) if class_names else None
    )
    
    return {
        "report": report,
        "confusion_matrix": cm,
        "raw_predictions": filtered_preds,
        "raw_labels": filtered_labels
    }