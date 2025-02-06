from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.error import URLError

import lightning as L
import torch
from torch import Tensor
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import VisionDataset
from torchvision.datasets.mnist import read_image_file, read_label_file
from torchvision.datasets.utils import check_integrity, download_and_extract_archive
from torchvision.transforms import v2 as v2


class MNIST(VisionDataset):
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

    @property
    def raw_folder(self) -> Path:
        return self.root.joinpath(self.__class__.__name__, "raw")

    @property
    def class_to_idx(self) -> dict[str, int]:
        return {_class: i for i, _class in enumerate(self.classes)}

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
            self.download()

        if not self._check_exists():
            raise RuntimeError(
                "Dataset not found. You can use download=True to download it"
            )

        self.data, self.targets = self._load_data()

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
        img, target = self.data[index], int(self.targets[index])

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def _check_exists(self) -> bool:
        return all(
            check_integrity(self.raw_folder.joinpath(Path(url).stem.split(".")[0]))
            for url, _ in self.resources
        )

    def download(self) -> None:
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


class MNISTDataModule(L.LightningDataModule):
    """lightning DataModule for MNIST dataset
    Ref: `https://lightning.ai/docs/pytorch/stable/data/datamodule.html#lightningdatamodule`
    """

    def __init__(
        self,
        data_dir: str | Path = "./data",
        batch_size: int = 32,
        transforms: v2.Compose | None = None,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size

        self.transform = transforms

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(
                mnist_full, [55000, 5000], generator=torch.Generator().manual_seed(42)
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.mnist_test = MNIST(
                self.data_dir, train=False, transform=self.transform
            )

    def train_dataloader(self) -> DataLoader:
        """This is the dataloader that the Trainer fit() method uses."""
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self) -> DataLoader:
        """This is the dataloader that the Trainer fit() and validate() methods uses."""
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        """This is the dataloader that the Trainer test() method uses."""
        return DataLoader(self.mnist_test, batch_size=self.batch_size)

    def prepare_data(self) -> None:
        """load raw data or tokenize data"""
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)
