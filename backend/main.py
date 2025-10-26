"""
Main FastAPI Application Entry Point
-----------------------------------
Initializes FastAPI app, includes routers, and sets up logging middleware.

Author: Willy Fitra Hendria
Last Updated: September 5, 2025
"""

from fastapi import FastAPI, Request, Response
from starlette.background import BackgroundTask

from loggers.logger import get_logger
from routers import api_router


logger = get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="MLOps Workflow",
    description="Modular MLOps Workflow.",
    version="0.1.0",
)

# Register unified API router
app.include_router(api_router.router, prefix="/api/v1")

# Prevent log messages from being propagated, thus avoid duplicated message
logger.propagate = False


@app.middleware("http")
async def log_middleware(request: Request, call_next):
    """
    Middleware to log each HTTP request and response.
    """
    response = await call_next(request)
    res_body = b""
    async for chunk in response.body_iterator:
        res_body += chunk
    task = BackgroundTask(
        log_request_response,
        request,
        response.status_code,
        res_body
    )
    return Response(
        content=res_body,
        status_code=response.status_code,
        headers=dict(response.headers),
        media_type=response.media_type,
        background=task
    )


def log_request_response(request, response_code, response_body):
    """
    Logs the HTTP request method, URL, response code, and response body.
    """
    logger.info(f"Request: {request.method} {request.url}")
    logger.info(
        f"Response code: {response_code}, Response body: "
        f"{response_body.decode('utf-8')}"
    )
