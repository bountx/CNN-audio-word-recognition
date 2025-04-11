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
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', epsilon=1e-7): # Add epsilon to init
        super(FocalLoss, self).__init__()
        if alpha is not None and not isinstance(alpha, torch.Tensor):
            alpha = torch.tensor(alpha, dtype=torch.float32)
        self.register_buffer('alpha', alpha)
        self.gamma = gamma
        self.reduction = reduction
        self.epsilon = epsilon # Store epsilon

    def forward(self, inputs, targets):
        num_classes = inputs.shape[1]
        log_probs = F.log_softmax(inputs, dim=1)
        log_p_t = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        p_t = torch.exp(log_p_t)

        # --- Add Epsilon for stability BEFORE calculating (1 - p_t) ---
        # Prevent p_t from being exactly 1.0 for the power calculation's gradient
        p_t_stable = p_t.clamp(min=self.epsilon, max=1.0 - self.epsilon)
        # Now use p_t_stable for the modulating factor calculation
        modulating_factor = (1 - p_t_stable) ** self.gamma
        # -------------------------------------------------------------

        # Calculate the focal loss term using the original log_p_t
        # We only needed p_t_stable for the modulating factor's gradient stability
        loss = -modulating_factor * log_p_t

        if self.alpha is not None:
            if self.alpha.shape[0] != num_classes:
                 raise ValueError(f"Alpha tensor size ({self.alpha.shape[0]}) must match number of classes ({num_classes})")
            alpha_t = self.alpha.gather(0, targets)
            loss = alpha_t * loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise ValueError(f"Invalid reduction type: {self.reduction}. Choose from 'none', 'mean', 'sum'.")