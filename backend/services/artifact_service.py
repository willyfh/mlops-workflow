"""
Artifact Service Module
----------------------
Handles model saving, registration, loading, caching, and MLflow signature.

Author: Willy Fitra Hendria
Last Updated: September 5, 2025
"""

from functools import lru_cache

import mlflow
import torch
from mlflow.exceptions import RestException

from conf.app_config import DEVICE, ENABLE_MODEL_CACHING, MAX_CACHE_SIZE
from services.messages import ClearCacheMessages, ModelRegistrationMessages
from schemas.result_schema import ClearCacheResult, ModelRegistrationResult


def save_model(model, artifact_path, signature, pip_requirements):
    """
    Save the model to the specified artifact path with the given signature and
    pip requirements file.

    Args:
    ----
        model (torch.nn.Module): The PyTorch model to be saved.
        artifact_path (str): The path where the model artifacts will be saved.
        signature (mlflow.models.signature.ModelSignature):
            The signature inferred by mlflow.
        pip_requirements (str):
            Path to the requirements.txt file for the model.

    Returns:
    -------
        mlflow.models.ModelRegistry.ModelVersion:
            Info about the registered model.

    """
    return mlflow.pytorch.log_model(
        model,
        artifact_path,
        signature=signature,
        pip_requirements=pip_requirements
    )


def register_model(run_id: str, model_name: str):
    """
    Register the model associated with the given run ID and model name.

    Args:
        run_id (str): The ID of the run containing the model to be registered.
        model_name (str): The name under which the model will be registered.

    Returns:
        ModelRegistrationResult: The result of the model registration process.
    """
    if run_id is None:
        run_id = _get_latest_run_id()

    if run_id is None:
        return ModelRegistrationResult(
            success=False,
            message=ModelRegistrationMessages.FAILURE.format(
                "Unable to retrieve latest run ID.",
            ),
        )

    result = mlflow.register_model(f"runs:/{run_id}/models", model_name)
    return ModelRegistrationResult(
        success=True,
        message=ModelRegistrationMessages.SUCCESS,
        run_id=str(run_id),
        model_name=result.name,
        model_version=result.version,
        model_aliases=result.aliases,
    )


def get_model_signature(model, input):
    """
    Generate and return the model signature using the input sample.

    Args:
    ----
        model (torch.nn.Module):
            The PyTorch model for which the signature will be generated.
        input (torch.Tensor):
            Input sample to be used for generating the signature.

    Returns:
    -------
        mlflow.models.signature.ModelSignature: The signature for the model.

    """
    model.eval()
    with torch.no_grad():
        output_sample = model(input)

    input_np = input.clone().detach().cpu().numpy()
    output_np = output_sample.clone().detach().cpu().numpy()
    signature = mlflow.models.infer_signature(input_np, output_np)
    return signature


def get_model(
    run_id: str = None,
    model_name: str = None,
    model_version: str = None,
    model_alias: str = None,
    device=DEVICE,
    enable_model_caching=ENABLE_MODEL_CACHING,
):
    """
    Retrieve and return the requested model, optionally caching it.

    Args:
    ----
        run_id (str, optional):
            ID of the run containing the model to retrieve.
        model_name (str, optional): The name of the model to retrieve.
        model_version (str, optional): The version of the model to retrieve.
        model_alias (str, optional): The alias of the model to retrieve.
        device (torch.device, optional): The device on which to load the model.
        enable_model_caching (bool, optional): Whether to enable model caching.

    Returns:
    -------
        torch.nn.Module: The requested model.

    """
    if enable_model_caching:
        model = _get_model_cached(
            run_id, model_name, model_version, model_alias, device,
        )
    else:
        model = _load_model(
            run_id, model_name, model_version, model_alias, device
        )
    return model


def clear_cache():
    """Clear the cache of cached models.

    Returns
    -------
        ClearCacheResult: The result of the cache clearing process.

    """
    _get_model_cached.cache_clear()
    return ClearCacheResult(success=True, message=ClearCacheMessages.SUCCESS)


@lru_cache(maxsize=MAX_CACHE_SIZE)  # Cache all results
def _get_model_cached(
    run_id: str, model_name: str, model_version: str, model_alias: str, device,
):
    """Cached version of get_model to avoid repetead download of the model."""
    model = _load_model(run_id, model_name, model_version, model_alias)
    return model


def _load_model(
    run_id: str = None,
    model_name: str = None,
    model_version: str = None,
    model_alias: str = None,
    device=DEVICE,
):
    """
    Load and return the requested model from MLflow.

    Args:
        run_id (str, optional): Run ID for the model.
        model_name (str, optional): Model name.
        model_version (str, optional): Model version.
        model_alias (str, optional): Model alias.
        device (torch.device, optional): Device to load the model on.

    Returns:
        torch.nn.Module: Loaded PyTorch model.
    """
    if run_id:
        model_uri = f"runs:/{run_id}/models"
    elif model_alias is not None:
        model_uri = f"models:/{model_name}@{model_alias}"
    else:
        model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.pytorch.load_model(model_uri, map_location=device)
    return model


def _get_latest_run_id():
    """Retrieve and return the latest run ID among all active experiments."""
    latest_run = mlflow.search_runs(
        order_by=["start_time DESC"],
        filter_string="tags.mlflow.runName = 'train'"
    )
    if not latest_run.empty:
        return latest_run.iloc[0]["run_id"]
    else:
        return None
