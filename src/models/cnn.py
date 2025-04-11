"""
CNN model architecture for speech recognition tasks.
"""

import numpy as np
import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self, num_mfcc_coeffs, fixed_length, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            # Input shape: (batch, 1, num_coeffs, fixed_length)
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 5), padding=(1, 2)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 5), padding=(1, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
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
            nn.Dropout(0.6),
            nn.Linear(128, num_classes)  # Output num_classes logits
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x  # Return raw logits