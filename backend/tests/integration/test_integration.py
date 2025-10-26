"""
Test Suite: Integration
----------------------
Covers integration tests for end-to-end MLOps Workflow: training, registration, and inference.

Author: Willy Fitra Hendria
Last Updated: September 5, 2025
"""

import base64

import mlflow
import pytest

from schemas.enums.enum import DatasetEnum, ModelTypeEnum
from services.messages import (
    InferenceMessages,
    ModelRegistrationMessages,
    TrainMessages,
)
from services.artifact_service import register_model
from services.inference_service import run_inference
from services.train_service import run_train

pytest_plugins = ("pytest_asyncio",)


@pytest.mark.asyncio()
async def test_integration(mlruns_dir):
    # Train a model
    train_config = {
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
    train_result = await run_train(train_config)
    assert train_result.success
    assert train_result.message == TrainMessages.SUCCESS

    # Register the trained model
    register_result = register_model(train_result.run_id, "dummy_model_name")
    assert register_result.success
    assert register_result.message == ModelRegistrationMessages.SUCCESS

    # Read image file for inference
    with open("tests/assets/3.png", "rb") as image_file:
        image_bytes = image_file.read()
    image_data = base64.b64encode(image_bytes)

    # Perform inference using the registered model
    inference_result = await run_inference(image_data, run_id=train_result.run_id)
    assert inference_result.success
    assert inference_result.message == InferenceMessages.SUCCESS
