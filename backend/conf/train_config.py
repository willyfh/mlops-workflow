"""
Training Configuration Schema
----------------------------
Defines Pydantic model for validating training parameters from Hydra config.

Author: Willy Fitra Hendria
Last Updated: September 5, 2025
"""

from pydantic import BaseModel, Field
from typing import Literal


class TrainConfig(BaseModel):
    """
    Pydantic schema for validating training configuration parameters.
    """
    max_epochs: int = Field(..., gt=0, description="Number of training epochs")
    lr: float = Field(..., gt=0, description="Learning rate")
    batch_size: int = Field(..., gt=0, description="Batch size")
    experiment_name: str = Field(
        ..., min_length=1,
        description="Experiment name"
    )
    model_type: Literal["SimpleCNN"] = Field(..., description="Model type")
    train_size: float = Field(
        ..., gt=0, le=1,
        description="Fraction of data for training"
    )
    dataset_name: Literal["MNIST"] = Field(..., description="Dataset name")
    seed: int = Field(..., description="Random seed")
