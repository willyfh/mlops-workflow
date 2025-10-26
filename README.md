# ⚙ MLOps Workflow

[![python 3.10](https://img.shields.io/badge/python-3.10-blue)](https://www.python.org/downloads/release/python-3100/)
[![python 3.11](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/downloads/release/python-3110/)
[![python 3.12](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/downloads/release/python-3120/)
[![Run Tests](https://github.com/willyfh/mlops-workflow/actions/workflows/ci-checks.yaml/badge.svg)](https://github.com/willyfh/mlops-workflow/actions/workflows/ci-checks.yaml)
[![codecov](https://codecov.io/gh/willyfh/mlops-workflow/graph/badge.svg?token=OGLCMT2KQ4)](https://codecov.io/gh/willyfh/mlops-workflow)
[![MIT License](https://opensource.org/licenses/MIT)](https://opensource.org/licenses/MIT)

A modular MLOps workflow for training, inference, experiment tracking, and deployment.
Built with FastAPI, MLflow, MinIO, and PostgreSQL for scalable machine learning operations.

## Features

- Supports concurrent (non-blocking) model training and inference requests via FastAPI
- MLflow for experiment tracking
- MinIO for artifact storage
- PostgreSQL for metadata and experiment storage
- Hydra for modular config
- Pydantic for config validation
- Docker Compose for easy deployment
- Pre-commit hooks for code quality
- Unit and integration tests

---

## Prerequisites

- Docker
- Nvidia container toolkit (optional, for GPU)
  - [Install Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

### GPU/CPU Configuration

- If you don’t want to use GPU, update your `.env` file:

  ```env
  NVIDIA_VISIBLE_DEVICES=
  NVIDIA_RUNTIME=
  ```

- If you want to use GPU:

  ```env
  NVIDIA_VISIBLE_DEVICES=all
  NVIDIA_RUNTIME=nvidia
  ```

---

## Installation & Usage

### Docker Compose

#### 1. Build and start containers

```sh
docker compose build
docker compose up
```

#### 2. Login to MinIO

[http://localhost:9001/login](http://localhost:9001/login)

- User: minioadmin
- Password: minioadmin (defined in `.env`)

#### 3. Create and copy access keys

- Create and copy the access keys
- Update `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` in `.env`

#### 4. Restart containers

```sh
docker compose up
```

#### 5. For training (from host)

```sh
curl -X POST -F "config_file=@backend/conf/config.yaml" \
  http://localhost:8000/api/v1/run_train
```

#### 6. For inference

`<run_id>` refers to the MLflow run ID

```sh
curl -X POST -H "Content-Type: application/json" -d '{
"image_data": "'"$(base64 -w 0 backend/tests/assets/3.png)"'"'
}' http://localhost:8000/api/v1/run_inference/run_id/<run_id>
```

---

## Service URLs

Here are the main web interfaces and endpoints you can access:

- **FastAPI API & Docs**
  - API root: [http://localhost:8000/api/v1/](http://localhost:8000/api/v1/)
  - Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
  - ReDoc: [http://localhost:8000/redoc](http://localhost:8000/redoc)

- **MLflow Tracking UI**
  - [http://localhost:5000](http://localhost:5000)

- **MinIO Console**
  - [http://localhost:9001/login](http://localhost:9001/login)

- **PostgreSQL**
  - Accessible via database clients at localhost:5432 (no web UI)

---

## Notes

- For GPU support, ensure Nvidia container toolkit is installed and configured.
- This project uses [uv](https://github.com/astral-sh/uv) as the Python dependency manager inside Docker containers

## Disclaimer

This repository is intended as a minimal, educational template or starter kit for machine learning workflows. The training logic and architecture are kept simple for clarity and ease of use. For production or research use, you are encouraged to extend and customize the code to fit your requirements.

## License

MIT
