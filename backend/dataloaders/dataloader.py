"""
Dataloaders Module
-----------------
Provides dataset loading, splitting, and transformation utilities for training and validation.

Author: Willy Fitra Hendria
Last Updated: September 5, 2025
"""

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset
from torchvision import datasets

from conf.app_config import DATASET_DIR
from schemas.enums.enum import DatasetEnum


class CustomSubset(Dataset):
    """
    Custom subset of torch dataset with transform.

    Args:
        subset (torch.utils.data.Subset): Subset of a torch dataset.
        transform (callable, optional): Transformation to apply to each sample.

    Methods:
        __getitem__(index): Returns transformed sample and label at index.
        __len__(): Returns the number of samples in the subset.
    """

    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        """
        Get a sample and label from the subset, applying transform if provided.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            tuple: (transformed sample, label)
        """
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        """
        Get the number of samples in the subset.

        Returns:
            int: Number of samples.
        """
        return len(self.subset)


def create_dataloader(
    dataset_name=DatasetEnum.MNIST.value,
    batch_size=64,
    train_size=0.9,
    train_transform=None,
    val_transform=None,
    seed=42,
    dataset_dir=DATASET_DIR,
):
    """Create dataloaders for training, validation, and testing datasets.

    Args:
    ----
        dataset_name (str): Name of the dataset. Defaults to MNIST.
        batch_size (int): Number of samples per batch. Defaults to 64.
        train_size (float | int): Size of the training dataset to include in the train split. Defaults to 0.9.
        train_transform (callable): A function/transform applied to the training dataset. Defaults to None.
        val_transform (callable): A function/transform applied to the validation dataset. Defaults to None.
        seed (int): Seed used for random splitting of the training dataset. Defaults to 42.
        dataset_dir (str): Directory where datasets are stored. Defaults to DATASET_DIR from app_config.

    Returns:
    -------
        tuple: A tuple containing three torch.utils.data.DataLoader objects:
            - train_loader: DataLoader for the training dataset.
            - val_loader: DataLoader for the validation dataset.
            - test_loader: DataLoader for the testing dataset.

    Raises:
    ------
        ValueError: If an unsupported dataset name is provided.

    """
    supported_datasets = ", ".join([dataset.value for dataset in DatasetEnum])

    if dataset_name == DatasetEnum.MNIST.value:
        train_dataset = datasets.MNIST(
            root=dataset_dir,
            train=True,
            download=True,
        )
        test_dataset = datasets.MNIST(
            root=dataset_dir,
            train=False,
            download=True,
            transform=val_transform,
        )

        # Splitting training dataset into train and validation
        train_indices, val_indices = train_test_split(
            range(len(train_dataset)),
            train_size=train_size,
            random_state=seed,
        )
        train_subset = Subset(train_dataset, train_indices)
        val_subset = Subset(train_dataset, val_indices)

        custom_train_subset = CustomSubset(train_subset, train_transform)
        custom_val_subset = CustomSubset(val_subset, val_transform)
    else:
        error_msg = f"Unsupported dataset name '{dataset_name}'. Supported dataset names are: {supported_datasets}."
        raise ValueError(error_msg)

    train_loader = torch.utils.data.DataLoader(
        custom_train_subset,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        custom_val_subset,
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader
