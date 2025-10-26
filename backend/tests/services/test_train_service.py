"""
Test Suite: Train Service
------------------------
Covers unit tests for training loop, experiment logging, and result validation.

Author: Willy Fitra Hendria
Last Updated: September 5, 2025
"""

import mlflow
import pytest

from schemas.enums.enum import DatasetEnum, ModelTypeEnum
from services.messages import TrainMessages
from services.train_service import run_train

pytest_plugins = "pytest_asyncio"


@pytest.mark.asyncio()
async def test_run_train(mlruns_dir):
    config = {
        "max_epochs": 1,
        "lr": 0.001,
        "batch_size": 64,
        "model_type": ModelTypeEnum.SIMPLE_CNN.value,
        "dataset_name": DatasetEnum.MNIST.value,
        "experiment_name": "dummy_experiment",
        "train_size": 0.8,
        "seed": 42,
    }
    mlflow.set_tracking_uri(mlruns_dir)

    result = await run_train(config)

    assert result.success
    assert result.message == TrainMessages.SUCCESS
