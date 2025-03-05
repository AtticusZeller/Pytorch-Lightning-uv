from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.error import URLError

import lightning as L
from torch import Tensor
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import VisionDataset
from torchvision.datasets.mnist import read_image_file, read_label_file
from torchvision.datasets.utils import check_integrity, download_and_extract_archive
from torchvision.transforms import v2 as v2


class DataSetBase(VisionDataset):
    classes: list[str]  # class = list[label]
    mean: tuple[float, ...]
    std: tuple[float, ...]

    def __init__(
        self,
        root: str | Path,
        train: bool = True,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        download: bool = True,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.root = Path(self.root)

        self.train = train  # training set or test set

        if download:
            self._download()

        if not self._check_exists():
            raise RuntimeError(
                "Dataset not found. You can use download=True to download it"
            )

        self.data, self.targets = self._load_data()

    def _load_data(self) -> tuple[Tensor, Tensor]:
        raise NotImplementedError

    def _download(self) -> None:
        raise NotImplementedError

    @property
    def raw_folder(self) -> Path:
        return self.root.joinpath(self.__class__.__name__, "raw")

    def __getitem__(self, index: int) -> tuple[Any | Tensor, Any | int]:
        img, target = self.data[index], int(self.targets[index])

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)


class MNIST(DataSetBase):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>` Dataset.
    Responsible for downloading,vectorize and splitting it into train and test dataset.
    It is a subclass of torch.utils.data Dataset class.
    It is necessary to override the ``__getitem__`` and ``__len__`` method.
    """

    mirrors = [
        "http://yann.lecun.com/exdb/mnist/",
        "https://ossci-datasets.s3.amazonaws.com/mnist/",
    ]

    resources = [
        ("train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
        ("train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
        ("t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
        ("t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c"),
    ]

    classes = [
        "0 - zero",
        "1 - one",
        "2 - two",
        "3 - three",
        "4 - four",
        "5 - five",
        "6 - six",
        "7 - seven",
        "8 - eight",
        "9 - nine",
    ]
    mean = (0.1307,)
    std = (0.3081,)

    def _load_data(self) -> tuple[Tensor, Tensor]:
        image_file = f"{'train' if self.train else 't10k'}-images-idx3-ubyte"
        data = read_image_file(self.raw_folder.joinpath(image_file))

        label_file = f"{'train' if self.train else 't10k'}-labels-idx1-ubyte"
        targets = read_label_file(self.raw_folder.joinpath(label_file))

        return data, targets

    def __getitem__(self, index: int) -> tuple[Any | Tensor, Any | int]:
        """get raw or transformed data

        Parameters
        ----------
        index : int

        Returns
        -------
        img: Tensor, shape(H,W)=28x28, dtype=torch.uint8
        target: int
        """
        return super().__getitem__(index)

    def _check_exists(self) -> bool:
        return all(
            check_integrity(self.raw_folder.joinpath(Path(url).stem.split(".")[0]))
            for url, _ in self.resources
        )

    def _download(self) -> None:
        # check
        if self._check_exists():
            return

        self.raw_folder.mkdir(parents=True, exist_ok=True)

        # download files
        for filename, md5 in self.resources:
            for mirror in self.mirrors:
                url = f"{mirror}{filename}"
                try:
                    print(f"Downloading {url}")
                    download_and_extract_archive(
                        url, download_root=self.raw_folder, filename=filename, md5=md5
                    )
                except URLError as error:
                    print(f"Failed to download (trying next):\n{error}")
                    continue
                finally:
                    print()
                break
            else:
                raise RuntimeError(f"Error downloading {filename}")

    def extra_repr(self) -> str:
        split = "Train" if self.train is True else "Test"
        return f"Split: {split}"


class FashionMNIST(MNIST):
    """`Fashion-MNIST <https://github.com/zalandoresearch/fashion-mnist>`_ Dataset."""

    mirrors = ["http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"]

    resources = [
        ("train-images-idx3-ubyte.gz", "8d4fb7e6c68d591d4c3dfef9ec88bf0d"),
        ("train-labels-idx1-ubyte.gz", "25c81989df183df01b3e8a0aad5dffbe"),
        ("t10k-images-idx3-ubyte.gz", "bef4ecab320f06d8554ea6380940ec79"),
        ("t10k-labels-idx1-ubyte.gz", "bb300cfdad3c16e7a12a480ee83cd310"),
    ]
    classes = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]
    mean = (0.2860,)
    std: tuple[float, ...] = (0.3530,)


class DataModule(L.LightningDataModule):
    """lightning DataModule for MNIST dataset
    Ref: `https://lightning.ai/docs/pytorch/stable/data/datamodule.html#lightningdatamodule`
    """

    def __init__(
        self,
        data: DataSetBase,
        data_dir: str | Path = "./data",
        batch_size: int = 32,
        transforms: v2.Compose | None = None,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size

        self.transform = transforms
        self.data = data
        self.num_workers = 8

    def setup(self, stage: str) -> None:
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            full_data = self.data(self.data_dir, train=True, transform=self.transform)
            self.train_data, self.val_data = random_split(full_data, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.test_data = self.data(
                self.data_dir, train=False, transform=self.transform
            )

    def train_dataloader(self) -> DataLoader:
        """This is the dataloader that the Trainer fit() method uses."""
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            pin_memory=True,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
            # drop_last=True,  # avoid compile error because of different batch size
        )

    def val_dataloader(self) -> DataLoader:
        """This is the dataloader that the Trainer fit() and validate() methods uses."""
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
            persistent_workers=True,
            # drop_last=True,  # avoid compile error because of different batch size
        )

    def test_dataloader(self) -> DataLoader:
        """This is the dataloader that the Trainer test() method uses."""
        return DataLoader(
            self.test_data,
            pin_memory=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def prepare_data(self) -> None:
        """load raw data or tokenize data"""
        self.data(self.data_dir, train=True, download=True)
        self.data(self.data_dir, train=False, download=True)
