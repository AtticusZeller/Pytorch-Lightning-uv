import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as v2

from pytorch_lightning_uv.data import (
    MNIST,
    MNISTDataModule,
    base_transform,
    reshape_image,
)


def test_dataset() -> None:
    train_dataset = MNIST(
        root="./data", train=True, download=True, transform=reshape_image
    )

    test_dataset = MNIST(
        root="./data", train=False, download=True, transform=reshape_image
    )
    train_img = train_dataset[1][0]
    test_img = test_dataset[1][0]

    assert torch.is_tensor(train_img)
    assert torch.is_tensor(test_img)
    assert len(train_dataset) == 60000
    assert len(test_dataset) == 10000
    assert train_img.shape == (1, 28, 28)
    assert test_img.shape == (1, 28, 28)


def test_MNISTDataModule():
    data_module = MNISTDataModule(
        data_dir="./data", batch_size=32, transforms=base_transform()
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
        break  # Test only first batch

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
        break  # Test only first batch

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
        break  # Test only first batch


def test_mean_std() -> None:
    train_dataset = MNIST(
        root="./data", train=True, download=True, transform=base_transform()
    )
    loader = DataLoader(train_dataset, batch_size=len(train_dataset))
    data = next(iter(loader))[0]
    mean, std = data.mean().item(), data.std().item()
    print("mean:", mean)  # 0.13066047430038452
    print("std:", std)  # 0.30810782313346863
