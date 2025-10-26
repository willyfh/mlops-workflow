
# Use official Python 3.12 slim image
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y git

# Install uv (Python dependency manager)
RUN pip install uv

# Set working directory
WORKDIR /mlflow

# Copy dependency file
COPY requirements.txt ./

# Install Python dependencies with uv
RUN uv pip install --system -r requirements.txt

# Copy rest of the code (if any)
COPY . .

# Command to run MLflow server
CMD ["sh", "-c", "mlflow server --backend-store-uri postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres/${POSTGRES_DB} --host 0.0.0.0 --serve-artifacts --artifacts-destination ${MLFLOW_ARTIFACTS_DESTINATION}"]
