"""
Models Module
-------------
Defines neural network architectures and model creation utilities for training and inference.

Author: Willy Fitra Hendria
Last Updated: September 5, 2025
"""

import torch
from torch import nn

from schemas.enums.enum import ModelTypeEnum


class SimpleCNN(nn.Module):
    """
    Simple Convolutional Neural Network (CNN) model.

    Architecture:
        - 2 convolutional layers
        - 1 max pooling layer
        - 2 fully connected layers
        - ReLU activations

    Methods:
        forward(x): Forward pass for input tensor x.
    """

    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        """
        Forward pass for the CNN model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 28, 28).

        Returns:
            torch.Tensor: Output logits for classification.
        """
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, kernel_size=2)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def create_model(model_type=ModelTypeEnum.SIMPLE_CNN.value):
    """
    Create a neural network model of the specified type.

    Args:
        model_type (str): Type of the model to create. Defaults to 'SimpleCNN'.

    Returns:
        torch.nn.Module: A neural network model instance.

    Raises:
        ValueError: If an unsupported model type is provided.
    """
    supported_models = ", ".join([model_type.value for model_type in ModelTypeEnum])
    if model_type == ModelTypeEnum.SIMPLE_CNN.value:
        return SimpleCNN()
    else:
        error_msg = (
            f"Unsupported model type '{model_type}'. Supported model types are: "
            f"{supported_models}."
        )
        raise ValueError(error_msg)
