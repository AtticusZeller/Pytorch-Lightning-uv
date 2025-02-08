"""
Ref:
    1. https://lightning.ai/docs/pytorch/stable/debug/debugging_basic.html
    2. https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch-lightning/Optimize_Pytorch_Lightning_models_with_Weights_%26_Biases.ipynb#scrollTo=h__ic9lC1saP
    3. https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelCheckpoint.html#lightning.pytorch.callbacks.ModelCheckpoint
"""

from dataclasses import asdict
from pathlib import Path

import torch
import typer
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar

from pytorch_lightning_uv.config import ConfigManager
from pytorch_lightning_uv.data.dataset import MNISTDataModule
from pytorch_lightning_uv.data.transform import train_transform
from pytorch_lightning_uv.eval.logger import LoggerManager
from pytorch_lightning_uv.model import MNIST_MLP
from pytorch_lightning_uv.utils import set_random_seed

set_random_seed()
torch.set_float32_matmul_precision("high")


def training(config_path: Path) -> None:
    # config
    config_manager = ConfigManager()
    config = config_manager.load_config(config_path)
    # dataset
    datamodule = MNISTDataModule(
        batch_size=config.data.batch_size, transforms=train_transform()
    )
    datamodule.prepare_data()
    datamodule.setup("fit")
    # model
    model = MNIST_MLP(
        n_layer_1=config.model.n_layer_1,
        n_layer_2=config.model.n_layer_2,
        lr=config.optimizer.lr,
    )
    # logger
    with LoggerManager(
        run_name=config.logger.run_name,
        entity=config.logger.entity,
        project=config.logger.project,
        config=asdict(config),
    ) as logger:
        logger.watch(model, log="all")
        # trainer
        checkpoint_callback = ModelCheckpoint(
            monitor="val_accuracy",
            mode="max",
            dirpath=f"checkpoints/{config.model.name}",
            filename=f"{config.model.name}-{config.data.dataset}"
            + "-{epoch:02d}-{val_accuracy:.2f}",
            save_top_k=1,
        )
        logger.after_save_checkpoint(checkpoint_callback)
        trainer = Trainer(
            logger=logger,
            callbacks=[RichProgressBar()],
            accelerator="gpu",
            max_epochs=config.training.max_epochs,
        )
        trainer.fit(model, datamodule)


def evaluation(config_path: Path, run_id: str) -> None:
    """Test the model from a specific wandb run.

    Args:
        config_path: Path to config file
        run_id: The wandb run ID to test (printed at end of training)
    """
    # config
    config_manager = ConfigManager()
    config = config_manager.load_config(config_path)
    # data
    datamodule = MNISTDataModule(
        batch_size=config.data.batch_size, transforms=train_transform()
    )
    datamodule.prepare_data()
    datamodule.setup("test")

    with LoggerManager(
        run_name=config.logger.run_name,
        entity=config.logger.entity,
        project=config.logger.project,
        id=run_id,
        config=asdict(config),
        job_type="eval",
    ) as logger:
        # model
        model_path = logger.load_best_model(run_id)
        model = MNIST_MLP.load_from_checkpoint(model_path)
        model.eval()
        # trainer
        trainer = Trainer(logger=logger, accelerator="gpu")
        results = trainer.test(model, datamodule)
        # log test results
        logger.log_test_results(results)


TRAIN_OPTION = typer.Option(False, "--train", help="Train the model")
EVAL_OPTION = typer.Option(False, "--eval", help="Test the model")
CONFIG_OPTION = typer.Option("data/train.yml", "-c", help="Path to config file")
RUN_ID_OPTION = typer.Option(None, "--run-id", help="WandB run ID to test")


def main(
    train: bool = TRAIN_OPTION,
    eval: bool = EVAL_OPTION,
    config: Path = CONFIG_OPTION,
    run_id: str | None = RUN_ID_OPTION,
) -> None:
    if train:
        training(config)
    elif eval:
        if run_id is None:
            print("Error: --run-id is required for testing")
            return
        evaluation(config, run_id)
    else:
        print("No command selected. Use --help to see available commands.")


if __name__ == "__main__":
    typer.run(main)
