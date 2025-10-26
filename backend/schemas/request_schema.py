"""
Request Schemas Module
---------------------
Defines Pydantic models for API request payloads:
inference and model registration requests.

Author: Willy Fitra Hendria
Last Updated: September 5, 2025
"""

from pydantic import BaseModel


class InferenceRequest(BaseModel):
    """
    Pydantic model for inference request.

    Attributes:
        image_data (str): Base64 encoded image data for inference.
        run_id (str, optional): Run identifier.
        model_name (str, optional): Name of the model to use.
        model_alias (str, optional): Alias of the model version.
    """

    image_data: str
    run_id: str | None = None
    model_name: str | None = None
    model_alias: str | None = None


class ModelRegistrationRequest(BaseModel):
    """
    Pydantic model for model registration request.

    Attributes:
        run_id (str, optional): Run identifier for the model.
        model_name (str): Name of the model to register.
    """

    run_id: str | None = None
    model_name: str
