"""Simple CNN model (PyTorch 60 Minute Blitz)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """Simple CNN model adapted from 'PyTorch: A 60 Minute Blitz'."""

    def __init__(self, num_classes: int = 10, num_channels: int = 3):
        """Initialize SimpleCNN.
        
        Args:
            num_classes: Number of output classes
            num_channels: Number of input channels (1 for grayscale, 3 for RGB)
        """
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        """Forward pass."""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# Alias for compatibility
Net = SimpleCNN
