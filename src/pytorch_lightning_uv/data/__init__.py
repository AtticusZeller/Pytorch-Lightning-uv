from torchvision.transforms import v2 as v2

from .dataset import MNIST, DataModule, FashionMNIST

__all__ = ["create_data_module"]


def create_data_module(
    name: str = "mnist", batch_size: int = 32, transforms: v2.Compose | None = None
) -> DataModule:
    data = {"mnist": MNIST, "fashion_mnist": FashionMNIST}[name.lower()]

    return DataModule(data, batch_size=batch_size, transforms=transforms)
