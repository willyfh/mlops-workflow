"""
Training Service Module
----------------------
Implements training loop, model evaluation, and MLflow experiment logging.

Author: Willy Fitra Hendria
Last Updated: September 5, 2025
"""

import asyncio
import datetime
import random
import psutil
import mlflow
import numpy as np
import torch
from hydra.utils import instantiate

from conf.app_config import DEVICE
from dataloaders.dataloader import create_dataloader
from loggers.logger import get_logger
from services.messages import TrainMessages
from models.model import create_model
from preprocess.preprocess import val_transform
from schemas.result_schema import TrainResult
from services.artifact_service import get_model_signature, save_model

logger = get_logger(__name__)


async def run_train(config_dict: dict) -> TrainResult:
    """
    Compose config with Hydra and run training. Used by both API and CLI.
    """
    from hydra import initialize, compose
    from omegaconf import OmegaConf
    conf_dir = "../conf"
    with initialize(config_path=conf_dir, version_base=None):
        overrides = [f"{k}={v}" for k, v in config_dict.items() if isinstance(v, (str, int, float))]
        cfg = compose(config_name="config", overrides=overrides)
        config = OmegaConf.to_container(cfg, resolve=True)

    def log_memory(stage: str):
        process = psutil.Process()
        mem_mb = process.memory_info().rss / 1024 / 1024
        logger.info(f"[MEMORY] {stage}: {mem_mb:.2f} MB used.")

    logger.info("[TRAIN] Starting run_train")
    log_memory("start")

    # Extract config values
    max_epochs = config.get("max_epochs")
    batch_size = config.get("batch_size")
    model_type = config.get("model_type")
    dataset_name = config.get("dataset_name")
    experiment_name = config.get("experiment_name")
    train_size = config.get("train_size")
    seed = config.get("seed")

    optimizer_cfg = config.get("optimizer", {})
    scheduler_cfg = config.get("scheduler", {})
    aug_cfg = config.get("augmentation", {})

    _set_seed(seed)

    mlflow_enabled = True
    try:
        experiment_id = _get_experiment_id(experiment_name)
    except Exception as e:
        logger.warning(f"MLflow server unavailable, skipping MLflow logging. Reason: {e}")
        mlflow_enabled = False
        experiment_id = None

    logger.info("[TRAIN] Instantiating augmentation pipeline via Hydra")
    log_memory("before dataloader")
    custom_train_transform = instantiate(aug_cfg)
    custom_val_transform = val_transform

    try:
        loop = asyncio.get_event_loop()
        logger.info("[TRAIN] Creating dataloaders...")
        train_loader, val_loader, test_loader = await loop.run_in_executor(
            None,
            create_dataloader,
            dataset_name,
            batch_size,
            train_size,
            custom_train_transform,
            custom_val_transform,
            seed
        )
        logger.info("[TRAIN] Dataloaders created successfully.")
        log_memory("after dataloader")
    except Exception as e:
        logger.error(f"[TRAIN] Dataloader creation failed: {e}")
        log_memory("dataloader exception")
        return TrainResult(
            success=False,
            message=TrainMessages.FAILURE.format(f"Dataloader creation failed: {e}"),
        )

    run_name = f"train_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger.info(config)
    logger.info("[TRAIN] Creating model...")
    log_memory("before model")
    try:
        model = create_model(model_type).to(DEVICE)
        logger.info("[TRAIN] Model created and moved to device.")
        log_memory("after model")
    except ValueError as e:
        logger.error(f"[TRAIN] Model creation failed: {e}")
        log_memory("model exception")
        return TrainResult(
            success=False, message=TrainMessages.FAILURE.format(str(e)),
        )

    logger.info("[TRAIN] Instantiating optimizer via Hydra")
    optimizer = instantiate(optimizer_cfg, model.parameters())
    logger.info("[TRAIN] Instantiating scheduler via Hydra (if any)")
    scheduler = instantiate(scheduler_cfg, optimizer) if scheduler_cfg.get("_target_") else None

    if mlflow_enabled:
        with mlflow.start_run(
            experiment_id=experiment_id,
            run_name=run_name
        ):
            mlflow.log_params(config)
            logger.info("[TRAIN] MLflow logging enabled.")
            logger.info("[TRAIN] Starting training loop...")
            log_memory("before training loop")
            model, metrics = await loop.run_in_executor(
                None,
                _train_model_ext,
                model,
                max_epochs,
                optimizer,
                scheduler,
                train_loader,
                val_loader,
                DEVICE
            )
            logger.info("[TRAIN] Training loop finished.")
            log_memory("after training loop")
            train_accuracy, val_accuracy, train_loss, val_loss = metrics

            logger.info("[TRAIN] Starting test evaluation...")
            log_memory("before test eval")
            test_loss, test_accuracy = await loop.run_in_executor(
                None, _test_model, model, test_loader, DEVICE,
            )
            logger.info("[TRAIN] Test evaluation finished.")
            log_memory("after test eval")

            logger.info("[TRAIN] Getting model signature...")
            sample_batch = next(iter(test_loader))
            input_sample = sample_batch[0].to(DEVICE)[:1]
            signature = await loop.run_in_executor(
                None, get_model_signature, model, input_sample,
            )
            logger.info("[TRAIN] Model signature obtained.")
            log_memory("after signature")

            result = save_model(
                model,
                "models",
                signature=signature,
                pip_requirements="requirements.txt"
            )
            logger.info("[TRAIN] Model saved.")
            log_memory("after save model")
    else:
        logger.info("[TRAIN] Running without MLflow logging.")
        logger.info("[TRAIN] Starting training loop...")
        log_memory("before training loop")
        model, metrics = await loop.run_in_executor(
            None,
            _train_model_ext,
            model,
            max_epochs,
            optimizer,
            scheduler,
            train_loader,
            val_loader,
            DEVICE
        )
        logger.info("[TRAIN] Training loop finished.")
        log_memory("after training loop")
        train_accuracy, val_accuracy, train_loss, val_loss = metrics

        logger.info("[TRAIN] Starting test evaluation...")
        log_memory("before test eval")
        test_loss, test_accuracy = await loop.run_in_executor(
            None, _test_model, model, test_loader, DEVICE,
        )
        logger.info("[TRAIN] Test evaluation finished.")
        log_memory("after test eval")

        logger.info("[TRAIN] Getting model signature...")
        sample_batch = next(iter(test_loader))
        input_sample = sample_batch[0].to(DEVICE)[:1]
        signature = await loop.run_in_executor(
            None, get_model_signature, model, input_sample,
        )
        logger.info("[TRAIN] Model signature obtained.")
        log_memory("after signature")

        result = save_model(
            model,
            "models",
            signature=signature,
            pip_requirements="requirements.txt"
        )
        logger.info("[TRAIN] Model saved.")
        log_memory("after save model")

    return TrainResult(
        success=True,
        message=TrainMessages.SUCCESS,
        run_id=result.run_id,
        train_accuracy=train_accuracy,
        val_accuracy=val_accuracy,
        test_accuracy=test_accuracy,
        train_loss=train_loss,
        val_loss=val_loss,
        test_loss=test_loss,
    )


def _train_model_ext(
    model,
    max_epochs,
    optimizer,
    scheduler,
    train_loader,
    val_loader,
    device=DEVICE,
    log_interval=50
):
    """
    Train the model using the specified optimizer, scheduler, and data loaders.

    Args:
        model (torch.nn.Module): Model to train.
        max_epochs (int): Number of training epochs.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Learning rate scheduler.
        train_loader (torch.utils.data.DataLoader): Training data loader.
        val_loader (torch.utils.data.DataLoader): Validation data loader.
        device (torch.device, optional): Device to train on. Defaults to DEVICE.
        log_interval (int, optional): Logging interval for training steps.

    Returns:
        tuple: (trained model, (train_accuracy, val_accuracy, train_loss, val_loss))
    """
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(max_epochs):
        # Training
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        for step, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted_train = torch.max(output.data, 1)
            total_train += target.size(0)
            correct_train += (predicted_train == target).sum().item()

            # Log metrics for each step
            if step % log_interval == 0 and step > 0:
                train_loss /= log_interval
                train_accuracy = (
                    correct_train / total_train if total_train > 0 else 0.0
                )
                logger.info(
                    f"Epoch {epoch+1}/{max_epochs}, "
                    f"Step {step}/{len(train_loader)}, "
                    f"Train Loss: {train_loss}, "
                    f"Train Accuracy: {train_accuracy}",
                )
                train_loss = 0.0
                correct_train = 0
                total_train = 0

        train_loss /= len(train_loader)
        train_accuracy = (
            correct_train / total_train if total_train > 0 else 0.0
        )

        # Validation
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()
                _, predicted_val = torch.max(output.data, 1)
                total_val += target.size(0)
                correct_val += (predicted_val == target).sum().item()
        val_loss /= len(val_loader)
        val_accuracy = correct_val / total_val if total_val > 0 else 0.0

        # Log metrics
        metrics = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy,
        }
        mlflow.log_metrics(metrics, step=epoch)

        logger.info(
            f"Epoch {epoch+1}/{max_epochs}, "
            f"Train Loss: {train_loss}, "
            f"Train Accuracy: {train_accuracy}, "
            f"Validation Loss: {val_loss}, "
            f"Validation Accuracy: {val_accuracy}",
        )

        # Step scheduler if present
        if scheduler is not None:
            scheduler.step()

    return model, (train_accuracy, val_accuracy, train_loss, val_loss)


def _test_model(model, test_loader, device=DEVICE):
    """
    Test the model using the specified test data loader.

    Args:
        model (torch.nn.Module): Model to test.
        test_loader (torch.utils.data.DataLoader): Test data loader.
        device (torch.device, optional): Device to test on. Defaults to DEVICE.

    Returns:
        tuple: (test_loss, test_accuracy)
    """
    logger.info("Testing the model on the test set...")
    model.eval()
    model = model.to(device)
    correct = 0
    total = 0
    test_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    test_loss /= len(test_loader)
    test_accuracy = correct / total if total > 0 else 0.0
    mlflow.log_metrics({"test_loss": test_loss, "test_accuracy": test_accuracy})
    logger.info(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
    return test_loss, test_accuracy


def _get_experiment_id(experiment_name):
    """
    Retrieve the experiment ID associated with the given experiment name.
    Creates a new experiment if it does not exist.

    Args:
        experiment_name (str): Name of the experiment.

    Returns:
        str: Experiment ID.
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(name=experiment_name)
    else:
        experiment_id = experiment.experiment_id
    return experiment_id


def _set_seed(seed):
    """
    Set the random seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
