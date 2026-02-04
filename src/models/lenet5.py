"""LeNet5 architecture for MNIST and similar datasets."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet5(nn.Module):
    """LeNet5 architecture."""
    
    def __init__(self, num_classes: int = 10, num_channels: int = 1):
        """Initialize LeNet5.
        
        Args:
            num_classes: Number of output classes
            num_channels: Number of input channels (1 for grayscale, 3 for RGB)
        """
        super(LeNet5, self).__init__()
        
        self.conv1 = nn.Conv2d(num_channels, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
    
    def forward(self, x):
        """Forward pass."""
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
