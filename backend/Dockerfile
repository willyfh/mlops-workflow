
# Use the official Python 3.12 slim image as base
FROM python:3.12-slim


# Install system dependencies
RUN apt-get update && apt-get install -y git gcc python3-dev


# Install uv (Python dependency manager)
RUN pip install uv

# Set the working directory in the container
WORKDIR /app


# Copy dependency files
COPY pyproject.toml requirements.txt ./


# Install Python dependencies with uv
RUN uv pip install --system -r requirements.txt


# Copy the rest of your code
COPY . .


# Expose port 8000
EXPOSE 8000

# Command to run the API
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "main:app", "-b", "0.0.0.0:8000", "-t", "0", "--workers", "3", "--threads", "4"]
