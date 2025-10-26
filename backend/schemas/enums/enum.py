"""
Enumerations Module
------------------
Defines enums for supported datasets and model types used throughout the framework.

Author: Willy Fitra Hendria
Last Updated: September 5, 2025
"""

from enum import Enum


class DatasetEnum(Enum):
    """
    Enumeration of supported datasets.

    Attributes:
        MNIST (str): MNIST dataset identifier.
    """

    MNIST = "MNIST"


class ModelTypeEnum(Enum):
    """
    Enumeration of model types.

    Attributes:
        SIMPLE_CNN (str): Simple CNN model identifier.
    """

    SIMPLE_CNN = "SimpleCNN"
