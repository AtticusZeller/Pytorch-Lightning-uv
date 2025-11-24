from collections.abc import Sequence

import torch
import torchvision.transforms.v2 as v2
from torchvision.transforms import InterpolationMode


def base_transform(target_size: tuple[int, int] = (224, 224)) -> v2.Compose:
    """
    Basic transform: convert to RGB, resize and convert to tensor.

    Parameters
    ----------
    target_size : tuple[int, int], default=(224, 224)
        Target size (height, width) for images

    Returns
    -------
    v2.Compose
        Composed transforms
    """
    return v2.Compose(
        [
            v2.ToImage(),
            v2.Grayscale(num_output_channels=3),  # Ensure all images have 3 channels
            v2.Resize(
                target_size, interpolation=InterpolationMode.BILINEAR, antialias=True
            ),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )


def standardize_transform(
    mean: Sequence[float],
    std: Sequence[float],
    target_size: tuple[int, int] = (224, 224),
) -> v2.Compose:
    """
    Transform with standardization using dataset statistics.

    Parameters
    ----------
    mean : Sequence[float]
        Mean values for each channel
    std : Sequence[float]
        Standard deviation values for each channel (expects 3 values for RGB)
    target_size : tuple[int, int], default=(224, 224)
        Target size (height, width) for images

    Returns
    -------
    v2.Compose
        Composed transforms
    """
    return v2.Compose(
        [
            v2.ToImage(),
            v2.Grayscale(num_output_channels=3),  # Ensure all images have 3 channels
            v2.Resize(
                target_size, interpolation=InterpolationMode.BILINEAR, antialias=True
            ),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std),
        ]
    )


def resnet_pt_transform() -> v2.Compose:
    """
    Transform for ResNet pretrained models (ImageNet preprocessing).

    Returns
    -------
    v2.Compose
        Composed transforms with ImageNet normalization
    """
    return v2.Compose(
        [
            v2.ToImage(),
            v2.Grayscale(num_output_channels=3),  # Ensure all images have 3 channels
            v2.Resize(
                235,
                interpolation=InterpolationMode.BICUBIC,
                max_size=None,
                antialias=True,
            ),
            v2.CenterCrop(224),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def efficientnetv2_pt_transform() -> v2.Compose:
    """
    Transform for EfficientNetV2 pretrained models.

    Returns
    -------
    v2.Compose
        Composed transforms
    """
    return v2.Compose(
        [
            v2.ToImage(),
            v2.Grayscale(num_output_channels=3),  # Ensure all images have 3 channels
            v2.Resize(
                300,
                interpolation=InterpolationMode.BICUBIC,
                max_size=None,
                antialias=True,
            ),
            v2.CenterCrop(300),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def to_tensor_transform() -> v2.Compose:
    """
    Transform to convert image to tensor.

    Returns
    -------
    v2.Compose
        Composed transforms
    """
    return v2.Compose([v2.ToImage()])
