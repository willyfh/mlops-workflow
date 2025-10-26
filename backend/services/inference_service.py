"""
Inference Service Module
-----------------------
Implements the main inference logic for serving model predictions via API.

Author: Willy Fitra Hendria
Last Updated: September 5, 2025
"""

import asyncio
import base64

import torch

from conf.app_config import DEVICE
from services.messages import InferenceMessages
from preprocess.preprocess import preprocess_image_bytes
from schemas.result_schema import InferenceResult
from services.artifact_service import get_model


async def run_inference(
    image_data: str,
    run_id: str | None = None,
    model_name: str | None = None,
    model_version: str | None = None,
    model_alias: str | None = None,
    device: torch.device = DEVICE,
) -> InferenceResult:
    """
    Run inference using the specified image data and model.

    Args:
        image_data (str): Base64 encoded image data.
        run_id (str, optional): Run identifier. Defaults to None.
        model_name (str, optional): Model name. Defaults to None.
        model_version (str, optional): Model version. Defaults to None.
        model_alias (str, optional): Model alias. Defaults to None.
        device (torch.device, optional): Device to run inference on. Defaults to DEVICE.

    Returns:
        InferenceResult: Inference result containing predicted class and probability.
    """
    # Check if run_id or model_name/version/alias are provided
    if not run_id and not (model_name and (model_version or model_alias)):
        return InferenceResult(
            success=False,
            message=InferenceMessages.FAILURE.format(
                "Either 'run_id' or both 'model_name' and 'model_version' (or 'model_alias') are not provided",
            ),
        )

    loop = asyncio.get_event_loop()

    try:
        model = await loop.run_in_executor(
            None,
            get_model,
            run_id,
            model_name,
            model_version,
            model_alias,
        )
    except ValueError as e:
        return InferenceResult(
            success=False,
            message=InferenceMessages.FAILURE.format(str(e)),
        )

    image_bytes = base64.b64decode(image_data)
    image_tensor = preprocess_image_bytes(image_bytes)
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        model = model.to(device)
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)[0]
        pred_class = torch.argmax(probabilities).item()
        probability = probabilities[pred_class].item()

    return InferenceResult(
        success=True,
        message=InferenceMessages.SUCCESS,
        pred_class=pred_class,
        probability=probability,
    )
