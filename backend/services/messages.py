"""
Messages Module
--------------
Defines message constants for API responses: registration, cache clearing,
inference, and training.

Author: Willy Fitra Hendria
Last Updated: September 5, 2025
"""


class ModelRegistrationMessages:
    """
    Message constants for model registration responses.
    """

    SUCCESS = "Model registered successfully."
    FAILURE = "Failed to register the model: {}"


class ClearCacheMessages:
    """
    Message constants for model cache clearing responses.
    """

    SUCCESS = "The cache of models has been cleared successfully."
    FAILURE = "Failed to clear the model cache: {}"


class InferenceMessages:
    """
    Message constants for inference responses.
    """

    SUCCESS = "The inference process was successful."
    FAILURE = "Failed to infer the prediction: {}"


class TrainMessages:
    """
    Message constants for training responses.
    """

    SUCCESS = "The training process has been finished successfully."
    FAILURE = "Failed to train the model: {}"
