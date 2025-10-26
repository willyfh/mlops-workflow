"""
Preprocessing Module
-------------------
Provides image preprocessing utilities and transformation pipelines for training and validation.

Author: Willy Fitra Hendria
Last Updated: September 5, 2025
"""

import io

from PIL import Image
from torchvision import transforms

# Define transformations for training
train_transform = transforms.Compose(
    [
        transforms.RandomRotation(15),  # Randomly rotate the image by up to 15 degrees
        transforms.RandomResizedCrop(28),  # Randomly crop a 28x28 portion of the image
        transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
        transforms.ToTensor(),  # Convert image to tensor
        transforms.Normalize((0.1307,), (0.3081,)),  # Normalize image
    ],
)

# Define transformations for validation
val_transform = transforms.Compose(
    [
        transforms.Resize((28, 28)),  # Resize the image to 28x28
        transforms.ToTensor(),  # Convert image to tensor
        transforms.Normalize((0.1307,), (0.3081,)),  # Normalize image
    ],
)


def preprocess_image_bytes(image_bytes, transform=val_transform):
    """
    Preprocess image bytes into a tensor suitable for model inference or training.

    Args:
        image_bytes (bytes): Image bytes to preprocess.
        transform (torchvision.transforms.Compose, optional): Transformation to apply. Defaults to val_transform.

    Returns:
        torch.Tensor: Preprocessed image tensor with batch dimension.
    """
    img = Image.open(io.BytesIO(image_bytes))

    image_tensor = transform(img).unsqueeze(0)  # Add batch dimension

    return image_tensor
