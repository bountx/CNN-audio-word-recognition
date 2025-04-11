# src/utils/loss.py (or directly in train.py)
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Implementation of Focal Loss for multi-class classification.
    Reference: "Focal Loss for Dense Object Detection" (https://arxiv.org/abs/1708.02002)

    Args:
        alpha (Tensor, optional): Weighting factor for each class. Can be a 1D tensor
                                  of size C (number of classes). If None, no class
                                  weighting is applied (equivalent to alpha=1 for all).
                                  Typically used to counterbalance class imbalance.
        gamma (float): Focusing parameter. Controls the rate at which easy examples
                       are down-weighted. gamma=0 recovers standard Cross Entropy.
                       Defaults to 2.0.
        reduction (str): Specifies the reduction to apply to the output:
                         'none' | 'mean' | 'sum'. 'mean' is common for training.
                         Defaults to 'mean'.
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        if alpha is not None and not isinstance(alpha, torch.Tensor):
             # Convert list or numpy array to tensor if needed
            alpha = torch.tensor(alpha, dtype=torch.float32)
        # Register alpha as a buffer to ensure it moves to the correct device
        self.register_buffer('alpha', alpha)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Model logits (raw outputs before softmax) - Shape (N, C)
            targets: True class indices - Shape (N,)
        Returns:
            Loss tensor based on the specified reduction.
        """
        num_classes = inputs.shape[1]
        
        # Calculate log probabilities using log_softmax for numerical stability
        log_probs = F.log_softmax(inputs, dim=1)

        # Gather the log probabilities corresponding to the true classes
        # targets need to be reshaped to (N, 1) for gather
        log_p_t = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1) # Shape: (N,)

        # Calculate the probabilities p_t
        p_t = torch.exp(log_p_t)

        # Calculate the modulating factor: (1 - p_t)^gamma
        modulating_factor = (1 - p_t) ** self.gamma

        # Calculate the focal loss term: - (1 - p_t)^gamma * log(p_t)
        loss = -modulating_factor * log_p_t

        # Apply alpha weighting (if alpha is specified)
        if self.alpha is not None:
            if self.alpha.shape[0] != num_classes:
                 raise ValueError(f"Alpha tensor size ({self.alpha.shape[0]}) must match number of classes ({num_classes})")
            # Gather the alpha weights for the true targets
            # Ensure alpha is on the same device as the loss tensor
            alpha_t = self.alpha.gather(0, targets) # Shape: (N,)
            loss = alpha_t * loss

        # Apply reduction method
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise ValueError(f"Invalid reduction type: {self.reduction}. Choose from 'none', 'mean', 'sum'.")