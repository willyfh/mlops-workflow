"""
Test Suite: Inference Service
----------------------------
Covers unit tests for model inference logic and API integration.

Author: Willy Fitra Hendria
Last Updated: September 5, 2025
"""

import base64
from unittest.mock import patch

import pytest
from torch import nn

from services.messages import InferenceMessages
from schemas.result_schema import InferenceResult
from services.inference_service import run_inference

pytest_plugins = ("pytest_asyncio",)


@pytest.mark.asyncio()
@patch("services.artifact_service._load_model")
async def test_run_inference(mock_get_model):
    mock_model = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=3),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Flatten(),
        nn.Linear(5408, 10),
    )
    # initizate model with a constant weight to avoid randomness (only for testing)
    for param in mock_model.parameters():
        param.data.fill_(0)

    mock_get_model.return_value = mock_model

    # Read image file in binary mode
    with open("tests/assets/3.png", "rb") as image_file:
        image_bytes = image_file.read()

    image_data = base64.b64encode(image_bytes)
    result = await run_inference(image_data, run_id="dummy_run_id")

    expected_result = InferenceResult(
        success=True,
        message=InferenceMessages.SUCCESS,
        pred_class=0,
        probability=0.10000000149011612,
    )
    assert result == expected_result
