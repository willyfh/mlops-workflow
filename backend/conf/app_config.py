"""
Application Configuration Module
--------------------------------
Centralizes global settings for dataset location, device, caching, and experiment parameters.

Author: Willy Fitra Hendria
Last Updated: September 5, 2025
"""


import torch
from schemas.enums.enum import DatasetEnum, ModelTypeEnum


# Dataset directory
DATASET_DIR = "./data"

# Inference caching configuration
ENABLE_MODEL_CACHING = True  # Enable/disable model caching for inference
MAX_CACHE_SIZE = 1           # Maximum number of models to cache

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Default experiment/training parameters
DEFAULT_EXPERIMENT_NAME = "default_experiment"
DEFAULT_BATCH_SIZE = 64
DEFAULT_LR = 0.000001
DEFAULT_MAX_EPOCHS = 1
DEFAULT_MODEL_TYPE = ModelTypeEnum.SIMPLE_CNN.value
DEFAULT_TRAIN_SIZE = 0.9  # Fraction of data for training
DEFAULT_DATASET_NAME = DatasetEnum.MNIST.value
DEFAULT_SEED = 42

# Default experiment config dictionary for quick access
DEFAULT_EXP_CONF_VAL = {
    "experiment_name": DEFAULT_EXPERIMENT_NAME,
    "max_epochs": DEFAULT_MAX_EPOCHS,
    "lr": DEFAULT_LR,
    "batch_size": DEFAULT_BATCH_SIZE,
    "model_type": DEFAULT_MODEL_TYPE,
    "train_size": DEFAULT_TRAIN_SIZE,
    "dataset_name": DEFAULT_DATASET_NAME,
    "seed": DEFAULT_SEED,
}
