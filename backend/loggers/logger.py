"""
Logger Module
-------------
Provides logger configuration and utilities for consistent logging across the project.

Author: Willy Fitra Hendria
Last Updated: September 5, 2025
"""

import logging


def get_logger(name):
    """
    Get a logger instance with the specified name.

    Configures the logger to log messages to the console at INFO level with a standard format.

    Args:
        name (str): The name for the logger.

    Returns:
        logging.Logger: A logger instance configured for console output.
    """
    # Create logger
    logger = logging.getLogger(name)

    # Configure logger
    logger.setLevel(logging.INFO)

    # Create console handler and set level to INFO
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Add formatter to console handler
    console_handler.setFormatter(formatter)

    # Add console handler to logger
    logger.addHandler(console_handler)

    return logger
