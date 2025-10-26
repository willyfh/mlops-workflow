"""
Result Schemas Module
--------------------
Defines Pydantic models for API responses: model registration, cache clearing,
inference, and training results.

Author: Willy Fitra Hendria
Last Updated: September 5, 2025
"""

from pydantic import BaseModel


class ModelRegistrationResult(BaseModel):
    """
    Pydantic model for model registration result.

    Attributes:
        success (bool): Indicates if registration was successful.
        message (str): Status or error message.
        run_id (str, optional): Run identifier.
        model_name (str, optional): Name of the registered model.
        model_version (int, optional): Version of the registered model.
        model_aliases (List[str], optional): Aliases for the model version.
    """

    success: bool
    message: str
    run_id: str | None = None
    model_name: str | None = None
    model_version: int | None = None
    model_aliases: list[str] | None = None


class ClearCacheResult(BaseModel):
    """
    Pydantic model for clear cache result.

    Attributes:
        success (bool): Indicates if cache clearing was successful.
        message (str): Status or error message.
    """

    success: bool
    message: str


class InferenceResult(BaseModel):
    """
    Pydantic model for inference result.

    Attributes:
        success (bool): Indicates if inference was successful.
        message (str): Status or error message.
        pred_class (int, optional): Predicted class index.
        probability (float, optional): Probability of the predicted class.
    """

    success: bool
    message: str
    pred_class: int | None = None
    probability: float | None = None


class TrainResult(BaseModel):
    """
    Pydantic model for training result.

    Attributes:
        success (bool): Indicates if training was successful.
        message (str): Status or error message.
        run_id (str, optional): Run identifier.
        train_accuracy (float, optional): Training accuracy.
        val_accuracy (float, optional): Validation accuracy.
        test_accuracy (float, optional): Test accuracy.
        train_loss (float, optional): Training loss.
        val_loss (float, optional): Validation loss.
        test_loss (float, optional): Test loss.
    """

    success: bool
    message: str
    run_id: str | None = None
    train_accuracy: float | None = None
    val_accuracy: float | None = None
    test_accuracy: float | None = None
    train_loss: float | None = None
    val_loss: float | None = None
    test_loss: float | None = None
