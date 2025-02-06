import numpy as np
import torch
import torchvision.transforms.v2 as v2
from kornia.image import ChannelsOrder, Image, ImageLayout, ImageSize, PixelFormat
from kornia.image.base import ColorSpace
from torch import Tensor, is_tensor
from torchvision import tv_tensors


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


def to_tensor(data: Image | np.ndarray | Tensor) -> Tensor:
    """Convert image tensor within [0,1]

    Parameters
    ----------
    img : Image | np.ndarray | Tensor

    Returns
    -------
    Tensor: dtype = torch.float32,value within [0,1]
    """
    if isinstance(data, Image):
        data = data.data

    if isinstance(data, np.ndarray):
        return torch.from_numpy(data).float().div(255)

    if is_tensor(data):
        if isinstance(data, torch.ByteTensor):
            return data.float().div(255)
        else:
            return data
    raise ValueError("Input should be either a kornia Image or a torch tensor")


def to_kornia_image(img: Tensor) -> Image:
    """Convert a torch tensor to a kornia Image.

    Parameters
    ----------
    img : Tensor
        Input tensor with shape (H, W)

    Returns
    -------
    Image
        Kornia Image with shape (1, H, W)
    """
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
    return Image(img_channels_first, pixel_format, layout)
