"""
Test Suite: Artifact Service
---------------------------
Covers unit tests for model registration, caching, and MLflow integration.

Author: Willy Fitra Hendria
Last Updated: September 5, 2025
"""

from unittest.mock import Mock, patch

import torch
from pandas import DataFrame
from torch import nn

from services.messages import ClearCacheMessages, ModelRegistrationMessages
from schemas.result_schema import ClearCacheResult, ModelRegistrationResult
from services.artifact_service import (
    clear_cache,
    get_model,
    get_model_signature,
    register_model,
)

mock_model1 = nn.Sequential(
    nn.Conv2d(3, 6, 5, padding="same"), nn.ReLU(), nn.MaxPool2d(2, 2),
)

mock_model2 = nn.Sequential(
    nn.Conv2d(3, 6, 5, padding="valid"), nn.ReLU(), nn.MaxPool2d(2, 2),
)


@patch("mlflow.register_model")
def test_register_model_with_run_id(mock_register_model):
    # Mock mlflow.register_model
    mock_result = Mock()
    mock_result.name = "dummy_model_name"
    mock_result.version = 1
    mock_result.aliases = ["alias"]
    mock_register_model.return_value = mock_result

    # Test successful registration with provided run_id
    result = register_model("dummy_run_id", "dummy_model_name")
    expected_result = ModelRegistrationResult(
        success=True,
        message=ModelRegistrationMessages.SUCCESS,
        run_id="dummy_run_id",
        model_name="dummy_model_name",
        model_version=1,
        model_aliases=["alias"],
    )
    assert result == expected_result


@patch("mlflow.register_model")
@patch("mlflow.search_runs")
def test_register_model_without_run_id(mock_search_runs, mock_register_model):
    # Mock mlflow.register_model
    mock_result = Mock()
    mock_result.name = "dummy_model_name"
    mock_result.version = 1
    mock_result.aliases = ["alias"]
    mock_register_model.return_value = mock_result

    mock_search_runs.side_effect = [
        DataFrame({"run_id": ["dummy_run_id"]}),
        DataFrame(),
    ]
    # Test registration with None run_id
    result1 = register_model(None, "dummy_model_name")
    result2 = register_model(None, "dummy_model_name")

    expected_result1 = ModelRegistrationResult(
        success=True,
        message=ModelRegistrationMessages.SUCCESS,
        run_id="dummy_run_id",
        model_name="dummy_model_name",
        model_version=1,
        model_aliases=["alias"],
    )
    expected_result2 = ModelRegistrationResult(
        success=False,
        message=ModelRegistrationMessages.FAILURE.format(
            "Unable to retrieve latest run ID.",
        ),
    )
    assert result1 == expected_result1
    assert result2 == expected_result2


def test_get_model_signature():
    # Create an example input tensor
    input_data = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image size

    # Test model signature
    signature = get_model_signature(mock_model1, input_data)

    assert str(signature.inputs) == "[Tensor('float32', (-1, 3, 32, 32))]"
    assert str(signature.outputs) == "[Tensor('float32', (-1, 6, 16, 16))]"


# Mock loading function with caching enabled
@patch("services.artifact_service._load_model")
def test_get_model_cached(mock_load_model):
    # Return different mock models on each call
    mock_load_model.side_effect = [mock_model1, mock_model2]

    # Test get_model with caching enabled
    result1 = get_model(run_id="dummy_run_id", enable_model_caching=True)
    result2 = get_model(run_id="dummy_run_id", enable_model_caching=True)

    assert result1 == mock_model1
    assert result2 == mock_model1  # same model should be returned with caching


# Mock loading function with caching disabled
@patch("services.artifact_service._load_model")
def test_get_model_load_different(mock_load_model):
    # Return different mock models on each call
    mock_load_model.side_effect = [mock_model1, mock_model2]

    # Test get_model with caching enabled
    result1 = get_model(run_id="dummy_run_id", enable_model_caching=False)
    result2 = get_model(run_id="dummy_run_id", enable_model_caching=False)

    assert result1 == mock_model1
    assert result2 == mock_model2  # same model should be returned with caching


def test_clear_cache():
    # Test clearing cache
    expected_result = ClearCacheResult(success=True, message=ClearCacheMessages.SUCCESS)
    assert clear_cache() == expected_result
