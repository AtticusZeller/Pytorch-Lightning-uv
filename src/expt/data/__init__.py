from typing import Literal

from torchvision.transforms import v2 as v2

from .dataset import MNIST, DataModule, FashionMNIST
from .transform import (
    base_transform,
    efficientnetv2_pt_transform,
    resnet_pt_transform,
    standardize_transform,
)

__all__ = ["create_data_module"]


def create_data_module(
    name: str = "mnist",
    batch_size: int = 32,
    transform: Literal[
        "standardize", "base", "resnet_pt", "efficientnetv2_pt"
    ] = "standardize",
) -> DataModule:
    data = {"mnist": MNIST, "fashion_mnist": FashionMNIST}[
        name.lower().replace(" ", "_")
    ]
    if transform == "standardize":
        return DataModule(
            data,
            batch_size=batch_size,
            transforms=standardize_transform(data.mean, data.std),
        )
    elif transform == "base":
        return DataModule(data, batch_size=batch_size, transforms=base_transform())
    elif transform == "resnet_pt":
        return DataModule(data, batch_size=batch_size, transforms=resnet_pt_transform())
    elif transform == "efficientnetv2_pt":
        return DataModule(
            data, batch_size=batch_size, transforms=efficientnetv2_pt_transform()
        )
    else:
        raise ValueError(f"Invalid transform type: {transform}")
