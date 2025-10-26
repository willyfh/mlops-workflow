"""
Test Suite: Models
------------------
Covers unit tests for model creation and validation logic.

Author: Willy Fitra Hendria
Last Updated: September 5, 2025
"""

import pytest

from schemas.enums.enum import ModelTypeEnum
from models.model import SimpleCNN, create_model


def test_create_model_simple_cnn():
    model = create_model(ModelTypeEnum.SIMPLE_CNN.value)
    assert isinstance(model, SimpleCNN)


def test_create_model_invalid_model_type():
    with pytest.raises(ValueError):
        create_model("InvalidModelType")
