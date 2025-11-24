from collections.abc import Callable
from pathlib import Path
from typing import Any

import lightning as L
from torch import Tensor
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import VisionDataset
from torchvision.datasets.mnist import read_image_file, read_label_file
from torchvision.datasets.utils import check_integrity
from torchvision.transforms import v2


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
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.root = Path(self.root)

        self.train = train  # training set or test set

        if not self._check_exists():
            raise RuntimeError(
                f"Dataset not found at {self.root}. "
                "Please ensure the dataset has been downloaded."
            )

        self.data, self.targets = self._load_data()

    def _load_data(self) -> tuple[Tensor, Tensor]:
        raise NotImplementedError

    def _check_exists(self) -> bool:
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

    Dataset should be downloaded using external scripts before use.
    This class only loads and processes pre-downloaded data.
    """

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
    mean = (0.1307, 0.1307, 0.1307)  # RGB channels for grayscale expanded to 3 channels
    std = (0.3081, 0.3081, 0.3081)

    def _load_data(self) -> tuple[Tensor, Tensor]:
        image_file = f"{'train' if self.train else 't10k'}-images-idx3-ubyte"
        data = read_image_file(self.raw_folder.joinpath(image_file).as_posix())

        label_file = f"{'train' if self.train else 't10k'}-labels-idx1-ubyte"
        targets = read_label_file(self.raw_folder.joinpath(label_file).as_posix())

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

    def extra_repr(self) -> str:
        split = "Train" if self.train is True else "Test"
        return f"Split: {split}"


class FashionMNIST(MNIST):
    """`Fashion-MNIST <https://github.com/zalandoresearch/fashion-mnist>`_ Dataset.

    Dataset should be downloaded using external scripts before use.
    """

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
    mean = (0.2860, 0.2860, 0.2860)  # RGB channels for grayscale expanded to 3 channels
    std: tuple[float, ...] = (0.3530, 0.3530, 0.3530)


class DataModule(L.LightningDataModule):
    """Lightning DataModule for datasets.

    Automatically initializes the dataset class based on data_dir path.
    Ref: `https://lightning.ai/docs/pytorch/stable/data/datamodule.html#lightningdatamodule`
    """

    # Dataset registry mapping directory names to dataset classes
    DATASET_REGISTRY = {"MNIST": MNIST, "FashionMNIST": FashionMNIST}

    def __init__(
        self,
        data_dir: str | Path = "./data",
        batch_size: int = 32,
        transforms: v2.Compose | None = None,
        val_split: float = 0.2,
    ) -> None:
        """
        Initialize DataModule.

        Parameters
        ----------
        data_dir : str | Path
            Root directory containing the dataset
        batch_size : int
            Batch size for dataloaders
        transforms : v2.Compose | None
            Transforms to apply to the data
        val_split : float
            Fraction of training data to use for validation
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.transform = transforms
        self.val_split = val_split
        self.num_workers = 8

        # Determine dataset class from directory structure
        self.dataset_class = self._get_dataset_class()

    def _get_dataset_class(self) -> type[DataSetBase]:
        """
        Determine which dataset class to use based on data_dir structure.

        Returns
        -------
        type[DataSetBase]
            Dataset class to instantiate

        Raises
        ------
        ValueError
            If dataset type cannot be determined from directory structure
        """
        # Check for dataset-specific subdirectories
        for dataset_name, dataset_cls in self.DATASET_REGISTRY.items():
            dataset_path = self.data_dir / dataset_name
            if dataset_path.exists():
                return dataset_cls

        raise ValueError(
            f"Cannot determine dataset type from {self.data_dir}. "
            f"Expected one of: {list(self.DATASET_REGISTRY.keys())}"
        )

    def setup(self, stage: str | None = None) -> None:
        """
        Setup datasets for different stages.

        Parameters
        ----------
        stage : str | None
            Stage name ('fit', 'validate', 'test', or None for all)
        """
        if stage == "fit" or stage is None:
            # Load full training dataset
            full_data = self.dataset_class(
                self.data_dir, train=True, transform=self.transform
            )

            # Calculate train/val split
            train_size = int(len(full_data) * (1 - self.val_split))
            val_size = len(full_data) - train_size
            self.train_data, self.val_data = random_split(
                full_data, [train_size, val_size]
            )

        if stage == "test" or stage is None:
            # Load test dataset
            self.test_data = self.dataset_class(
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
        )

    def val_dataloader(self) -> DataLoader:
        """This is the dataloader that the Trainer fit() and validate() methods uses."""
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
            persistent_workers=True,
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
        """
        Prepare data (download, tokenize, etc.).

        This method is called only on one GPU in distributed training.
        For this dataset, we just verify it exists.
        """
        if not self.data_dir.exists():
            raise RuntimeError(
                f"Dataset directory not found: {self.data_dir}\n"
                "Please ensure the dataset has been downloaded using the provided scripts."
            )
