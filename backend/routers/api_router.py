"""
Unified API Router Module
------------------------
Includes all FastAPI endpoints for training, inference, and artifact management.

Author: Willy Fitra Hendria
Last Updated: September 5, 2025
"""

from fastapi import APIRouter, File, UploadFile, BackgroundTasks
from schemas.request_schema import (
    ModelRegistrationRequest,
    InferenceRequest
)
from schemas.result_schema import (
    ClearCacheResult,
    ModelRegistrationResult,
    InferenceResult
)
from services import artifact_service, inference_service, train_service
import yaml

router = APIRouter()


@router.post("/register_model/latest", response_model=ModelRegistrationResult)
async def register_model_latest(request_body: ModelRegistrationRequest):
    return artifact_service.register_model(None, request_body.model_name)


@router.post("/register_model/run_id/{run_id}", response_model=ModelRegistrationResult)
async def register_model_with_run_id(request_body: ModelRegistrationRequest):
    return artifact_service.register_model(
        request_body.run_id, request_body.model_name
    )


@router.post("/clear_cached_model", response_model=ClearCacheResult)
async def clear_cached_model():
    return artifact_service.clear_cache()


@router.post(
    "/run_inference/model/{model_name}/version/{model_version}",
    response_model=InferenceResult
)
async def run_inference_with_model_name_and_model_version(
    model_name: str, model_version: int, request_body: InferenceRequest
):
    image_data = request_body.image_data
    return await inference_service.run_inference(
        image_data=image_data, model_name=model_name, model_version=model_version
    )


@router.post(
    "/run_inference/model/{model_name}/alias/{model_alias}",
    response_model=InferenceResult
)
async def run_inference_with_model_name_and_model_alias(
    model_name: str, model_alias: str, request_body: InferenceRequest
):
    image_data = request_body.image_data
    return await inference_service.run_inference(
        image_data=image_data, model_name=model_name, model_alias=model_alias
    )


@router.post("/run_inference/run_id/{run_id}", response_model=InferenceResult)
async def run_inference_with_run_id(run_id: str, request_body: InferenceRequest):
    image_data = request_body.image_data
    return await inference_service.run_inference(
        image_data=image_data, run_id=run_id
    )


def background_train(config_dict):
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(train_service.run_train(config_dict))
    loop.close()


@router.post("/run_train")
async def run_train(
    background_tasks: BackgroundTasks, config_file: UploadFile = File(...)
):
    config_bytes = await config_file.read()
    config = yaml.safe_load(config_bytes)
    background_tasks.add_task(background_train, config)
    return {"success": True, "message": "Training started in background."}
