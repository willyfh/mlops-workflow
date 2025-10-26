"""
Training Script
---------------
Entry point for running model training using Hydra and Pydantic config.

Author: Willy Fitra Hendria
Last Updated: September 5, 2025
"""

import asyncio
import os
from conf.train_config import TrainConfig
from loggers.logger import get_logger
from services.train_service import run_train


def main():
    # Load config using Hydra YAML or Pydantic, then pass as dict to service
    import yaml
    from pathlib import Path
    logger = get_logger(__name__)
    script_dir = Path(__file__).parent
    default_config_path = script_dir / "conf" / "config.yaml"
    config_path = os.environ.get("CONFIG_PATH", str(default_config_path))
    with open(config_path, "r") as f:
        config_yaml = yaml.safe_load(f)
    # Optionally validate with Pydantic
    config = TrainConfig(**config_yaml)

    mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if mlflow_tracking_uri is None:
        logger.warning(
            "MLFLOW_TRACKING_URI not set. Defaulting to http://localhost:5000."
        )
        mlflow_tracking_uri = "http://localhost:5000"
        os.environ["MLFLOW_TRACKING_URI"] = mlflow_tracking_uri

    result = asyncio.run(run_train(config.dict()))
    print(result)


if __name__ == "__main__":
    main()
