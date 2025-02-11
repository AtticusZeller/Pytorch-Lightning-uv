from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from torch import Tensor
from torch.utils.data import DataLoader

from pytorch_lightning_uv.config import ConfigManager
from pytorch_lightning_uv.data import create_data_module
from pytorch_lightning_uv.data.transform import base_transform
from pytorch_lightning_uv.eval.logger import LoggerManager
from pytorch_lightning_uv.utils import set_random_seed


def label_distribution(
    train_loader: DataLoader[tuple[Tensor, Tensor]],
    val_loader: DataLoader[tuple[Tensor, Tensor]],
    test_loader: DataLoader[tuple[Tensor, Tensor]],
    logger_manager: LoggerManager,
) -> None:
    """Log label distributions"""
    for name, loader in [
        ("train", train_loader),
        ("val", val_loader),
        ("test", test_loader),
    ]:
        # Collect labels
        all_labels = []
        for _, labels in loader:
            all_labels.extend(labels.numpy())

        # Count label frequencies
        label_counts = np.bincount(np.array(all_labels))

        # Create distribution plot
        plt.figure(figsize=(10, 6))
        plt.bar(range(10), label_counts)
        plt.title(f"{name} Set Label Distribution")
        plt.xlabel("Digit")
        plt.ylabel("Count")

        # Log to wandb
        logger_manager.log_image(f"distribution/{name}", [plt.gcf()])
        plt.close()

        # Log as table
        logger_manager.log_table(
            key=f"stats/{name}_distribution",
            columns=["digit", "count"],
            data=[[i, count] for i, count in enumerate(label_counts)],
        )


def sample_images(
    train_loader: DataLoader[tuple[Tensor, Tensor]], logger_manager: LoggerManager
) -> None:
    # Log sample images
    for batch_idx, (images, labels) in enumerate(train_loader):
        if batch_idx == 0:
            # Get first 25 images
            sample_images = images[:25]
            sample_labels = labels[:25]

            # Create grid plot
            fig, axes = plt.subplots(5, 5, figsize=(10, 10))
            for idx, (img, label) in enumerate(
                zip(sample_images, sample_labels, strict=True)
            ):
                ax = axes[idx // 5, idx % 5]
                ax.imshow(img[0], cmap="gray")
                ax.set_title(f"Label: {label}")
                ax.axis("off")
            plt.tight_layout()

            # Log to wandb
            logger_manager.log_image(
                "samples/grid", [plt.gcf()], caption=["Sample MNIST digits"]
            )
            plt.close()

            # Also log individual images with matching number of captions
            images_to_log = [img[0].numpy() for img in sample_images]
            logger_manager.log_image(
                "samples/individual",
                images_to_log,
                caption=[f"Digit {label.item()}" for label in sample_labels],
            )
            break


def analyze_mnist_with_wandb(name: str = "mnist") -> None:
    """Analyze MNIST dataset using Weights & Biases logging"""
    config_manager = ConfigManager()
    config = config_manager.load_config(Path("./config/train.yml"))
    config.logger.run_name = "explore dataset analysis"

    # Initialize logger
    with LoggerManager(
        run_name=config.logger.run_name,
        entity=config.logger.entity,
        project=config.logger.project,
        job_type="eval",
        config=config,
    ) as logger_manager:
        # Initialize data module
        data_module = create_data_module(
            name=config.data.dataset,
            batch_size=config.data.batch_size,
            transforms=base_transform(),
        )

        # Prepare and setup data
        data_module.prepare_data()
        data_module.setup("fit")
        data_module.setup("test")

        # Get dataloaders
        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()
        test_loader = data_module.test_dataloader()

        label_distribution(train_loader, val_loader, test_loader, logger_manager)
        sample_images(train_loader, logger_manager)


if __name__ == "__main__":
    set_random_seed()
    analyze_mnist_with_wandb()
