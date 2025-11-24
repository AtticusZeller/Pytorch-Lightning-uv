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
    data_dir: str = "./data",
    batch_size: int = 32,
    transform: Literal[
        "standardize", "base", "resnet_pt", "efficientnetv2_pt"
    ] = "standardize",
) -> DataModule:
    """
    Create a DataModule for the specified dataset.

    Parameters
    ----------
    name : str
        Dataset name ("mnist" or "fashion_mnist")
    data_dir : str
        Root directory containing the dataset
    batch_size : int
        Batch size for dataloaders
    transform : Literal
        Transform type to apply

    Returns
    -------
    DataModule
        Configured DataModule instance
    """
    # Get dataset class to access mean/std for standardization
    dataset_classes = {"mnist": MNIST, "fashion_mnist": FashionMNIST}
    dataset_cls = dataset_classes[name.lower().replace(" ", "_")]

    if transform == "standardize":
        return DataModule(
            data_dir=data_dir,
            batch_size=batch_size,
            transforms=standardize_transform(dataset_cls.mean, dataset_cls.std),
        )
    elif transform == "base":
        return DataModule(
            data_dir=data_dir, batch_size=batch_size, transforms=base_transform()
        )
    elif transform == "resnet_pt":
        return DataModule(
            data_dir=data_dir, batch_size=batch_size, transforms=resnet_pt_transform()
        )
    elif transform == "efficientnetv2_pt":
        return DataModule(
            data_dir=data_dir,
            batch_size=batch_size,
            transforms=efficientnetv2_pt_transform(),
        )
    else:
        raise ValueError(f"Invalid transform type: {transform}")
