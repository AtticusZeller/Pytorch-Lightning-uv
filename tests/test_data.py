import torch
from kornia.image import Image

from pytorch_lightning_uv.data import MNIST, to_kornia_image


def test_dataset():
    train_dataset = MNIST(
        root="./data", train=True, download=True, transform=to_kornia_image
    )

    test_dataset = MNIST(
        root="./data", train=False, download=True, transform=to_kornia_image
    )
    train_img = train_dataset[1][0]
    test_img = test_dataset[1][0]

    assert isinstance(train_img, Image)
    assert isinstance(test_img, Image)
    assert len(train_dataset) == 60000
    assert len(test_dataset) == 10000
    assert train_img.shape == (1, 28, 28)
    assert test_img.shape == (1, 28, 28)
    assert train_img.dtype == torch.uint8
    assert test_img.dtype == torch.uint8
