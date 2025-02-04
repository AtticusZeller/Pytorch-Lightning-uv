import torch
from kornia.image import Image
from torchvision import transforms

from pytorch_lightning_uv.data import MNIST, to_kornia_image, to_torch_tensor


def test_dataset() -> None:
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


def test_dataloader() -> None:
    from torch.utils.data import DataLoader

    transform_funcs = transforms.Compose([to_kornia_image, to_torch_tensor])

    train_dataset = MNIST(
        root="./data", train=True, download=True, transform=transform_funcs
    )

    batch_size = 32
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    assert len(train_loader) == len(train_dataset) // batch_size

    sample_batch = next(iter(train_loader))
    images, labels = sample_batch

    assert torch.is_tensor(images)
    assert images.shape == (batch_size, 1, 28, 28)
    assert images.dtype == torch.uint8

    assert labels.shape == (batch_size,)
    assert labels.dtype == torch.int64

    for batch_idx, (batch_images, batch_labels) in enumerate(train_loader):
        assert batch_images.shape == (batch_size, 1, 28, 28)
        assert batch_labels.shape == (batch_size,)
        if batch_idx == 1:  # 测试前2个批次即可
            break

    test_loader = DataLoader(
        MNIST(root="./data", train=False, transform=transform_funcs),
        batch_size=batch_size,
        num_workers=4,
    )

    test_images, test_labels = next(iter(test_loader))
    assert test_images.shape == (batch_size, 1, 28, 28)
    assert test_labels.shape == (batch_size,)
