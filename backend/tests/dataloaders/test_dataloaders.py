"""
Test Suite: Dataloaders
----------------------
Covers unit tests for dataloader creation and dataset handling.

Author: Willy Fitra Hendria
Last Updated: September 5, 2025
"""

import shutil

import pytest
import torch
from torchvision import transforms

from dataloaders.dataloader import create_dataloader
from schemas.enums.enum import DatasetEnum


@pytest.fixture()
def transforms_config():
    train_transform = transforms.Compose([transforms.ToTensor()])
    val_transform = transforms.Compose([transforms.ToTensor()])
    return train_transform, val_transform


@pytest.fixture()
def dataset_dir(tmp_path):
    # Create the directory
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    yield data_dir  # This yields the temporary dataset directory
    # Clean up after the test is done
    shutil.rmtree(data_dir)


def test_create_dataloader_mnist(transforms_config, dataset_dir):
    train_transform, val_transform = transforms_config
    train_loader, val_loader, test_loader = create_dataloader(
        dataset_name=DatasetEnum.MNIST.value,
        train_transform=train_transform,
        val_transform=val_transform,
        dataset_dir=dataset_dir,
    )

    assert len(train_loader.dataset) == 54000  # 90% of 60000 training samples
    assert len(val_loader.dataset) == 6000  # 10% of 60000 training samples
    assert len(test_loader.dataset) == 10000  # Test dataset size for MNIST
    assert train_loader.batch_size == 64
    assert val_loader.batch_size == 64
    assert test_loader.batch_size == 64
    assert next(iter(train_loader))[0].shape == torch.Size(
        [64, 1, 28, 28],
    )  # Sample batch shape


def test_create_dataset_invalid_dataset_name():
    with pytest.raises(ValueError):
        create_dataloader("InvalidDatasetName")
