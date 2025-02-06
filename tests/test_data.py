import pytest
import torch
from kornia.image import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from pytorch_lightning_uv.data import MNIST, MNISTDataModule, to_kornia_image, to_tensor


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
    transform_funcs = transforms.Compose([to_kornia_image, to_tensor])

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
    assert images.dtype == torch.float32

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


def test_to_tensor() -> None:
    # Test with kornia Image
    img_tensor = torch.randint(0, 255, (28, 28), dtype=torch.uint8)
    kornia_img = to_kornia_image(img_tensor)
    result = to_tensor(kornia_img)
    assert torch.is_tensor(result)
    assert result.dtype == torch.float32
    assert result.max() <= 1.0
    assert result.min() >= 0.0

    # Test with numpy array
    np_img = torch.randint(0, 255, (28, 28)).numpy()
    result = to_tensor(np_img)
    assert torch.is_tensor(result)
    assert result.dtype == torch.float32
    assert result.max() <= 1.0
    assert result.min() >= 0.0

    # Test with ByteTensor
    byte_tensor = torch.randint(0, 255, (28, 28), dtype=torch.uint8)
    result = to_tensor(byte_tensor)
    assert torch.is_tensor(result)
    assert result.dtype == torch.float32
    assert result.max() <= 1.0
    assert result.min() >= 0.0

    # Test with float Tensor
    float_tensor = torch.rand(28, 28)
    result = to_tensor(float_tensor)
    assert torch.is_tensor(result)
    assert result.dtype == torch.float32
    assert torch.equal(result, float_tensor)

    # Test invalid input
    with pytest.raises(ValueError):
        to_tensor("invalid input")


def test_MNISTDataModule():
    data_module = MNISTDataModule(
        data_dir="./data", batch_size=32, transforms=[to_kornia_image, to_tensor]
    )
    data_module.prepare_data()
    data_module.setup("fit")
    data_module.setup("test")
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()
    # Verify train loader properties and data
    assert isinstance(train_loader, DataLoader)
    assert len(train_loader.dataset) == 55000
    for batch in train_loader:
        images, labels = batch
        assert images.shape[0] == min(32, len(images))
        assert images.shape[1:] == (1, 28, 28)
        assert images.dtype == torch.float32
        assert labels.shape[0] == min(32, len(labels))
        assert labels.dtype == torch.int64
        assert torch.max(images) <= 1.0
        assert torch.min(images) >= 0.0
        assert torch.max(labels) < 10
        assert torch.min(labels) >= 0

    # Verify val loader properties and data
    assert isinstance(val_loader, DataLoader)
    assert len(val_loader.dataset) == 5000
    for batch in val_loader:
        images, labels = batch
        assert images.shape[0] == min(32, len(images))
        assert images.shape[1:] == (1, 28, 28)
        assert images.dtype == torch.float32
        assert labels.shape[0] == min(32, len(labels))
        assert labels.dtype == torch.int64
        assert torch.max(images) <= 1.0
        assert torch.min(images) >= 0.0
        assert torch.max(labels) < 10
        assert torch.min(labels) >= 0

    # Verify test loader properties and data
    assert isinstance(test_loader, DataLoader)
    assert len(test_loader.dataset) == 10000
    for batch in test_loader:
        images, labels = batch
        assert images.shape[0] == min(32, len(images))
        assert images.shape[1:] == (1, 28, 28)
        assert images.dtype == torch.float32
        assert labels.shape[0] == min(32, len(labels))
        assert labels.dtype == torch.int64
        assert torch.max(images) <= 1.0
        assert torch.min(images) >= 0.0
        assert torch.max(labels) < 10
        assert torch.min(labels) >= 0


def test_mean_std() -> None:
    transform_funcs = transforms.Compose([to_kornia_image, to_tensor])
    train_dataset = MNIST(
        root="./data", train=True, download=True, transform=transform_funcs
    )
    loader = DataLoader(train_dataset, batch_size=len(train_dataset))
    data = next(iter(loader))[0]
    mean, std = data.mean().item(), data.std().item()
    print("mean:", mean)  # 0.13066047430038452
    print("std:", std)  # 0.30810782313346863
