from collections.abc import Sequence

import torch
import torchvision.transforms.v2 as v2
from kornia.image import ChannelsOrder, Image, ImageLayout, ImageSize, PixelFormat
from kornia.image.base import ColorSpace
from torch import Tensor
from torchvision.transforms import InterpolationMode

# def create_augmentation_pipeline(image_size=224) -> v2.Compose:
#     """
#     Creates data augmentation pipeline for object detection
#     """
#     transforms = v2.Compose(
#         [
#             # Convert input to standard format
#             v2.ToImage(),
#             # Color augmentations
#             v2.RandomPhotometricDistort(
#                 brightness=(0.8, 1.2),
#                 contrast=(0.8, 1.2),
#                 saturation=(0.8, 1.2),
#                 hue=(-0.1, 0.1),
#                 p=1.0,
#             ),
#             # Geometric augmentations
#             v2.RandomZoomOut(
#                 fill={tv_tensors.Image: (123, 117, 104), "others": 0},
#                 side_range=(1.0, 4.0),
#                 p=0.5,
#             ),
#             v2.RandomIoUCrop(),
#             v2.RandomHorizontalFlip(p=1.0),
#             # Clean up bounding boxes
#             v2.SanitizeBoundingBoxes(),
#             # Convert to final format
#             v2.ToDtype(torch.float32, scale=True),
#         ]
#     )
#     return transforms


def reshape_image(img: Tensor) -> Tensor:
    """Reshape image tensor to channels first

    Parameters
    ----------
    img : Tensor
        Input tensor with shape (H, W)

    Returns
    -------
    Image
        Kornia Image Tensor with shape (C, H, W)
    """
    if img.dim() == 2:
        # Add channel dimension
        img_channels_first = img.unsqueeze(0)

    # Define image layout
    layout = ImageLayout(
        image_size=ImageSize(28, 28),
        channels=1,
        channels_order=ChannelsOrder.CHANNELS_FIRST,
    )

    # Define pixel format
    pixel_format = PixelFormat(color_space=ColorSpace.GRAY, bit_depth=8)

    # Create kornia Image
    return Image(img_channels_first, pixel_format, layout).data


def base_transform() -> v2.Compose:
    return v2.Compose([reshape_image, v2.ToDtype(torch.float32, scale=True)])


def standardize_transform(mean: Sequence[float], std: Sequence[float]) -> v2.Compose:
    return v2.Compose(
        [reshape_image, v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean, std)]
    )


def resnet_pt_transform() -> v2.Compose:
    return v2.Compose(
        [
            reshape_image,
            v2.Grayscale(num_output_channels=3),
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
    return v2.Compose(
        [
            reshape_image,
            v2.Grayscale(num_output_channels=3),
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
