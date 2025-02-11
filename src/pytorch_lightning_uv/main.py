"""
Ref:
    1. https://lightning.ai/docs/pytorch/stable/debug/debugging_basic.html
    2. https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch-lightning/Optimize_Pytorch_Lightning_models_with_Weights_%26_Biases.ipynb#scrollTo=h__ic9lC1saP
    3. https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelCheckpoint.html#lightning.pytorch.callbacks.ModelCheckpoint
"""

from pathlib import Path

import torch
import typer
from lightning import Trainer

import wandb
from pytorch_lightning_uv.cli import (
    ConfigPath,
    EDAFlag,
    EvalFlag,
    RunID,
    SweepConfigPath,
    SweepCount,
    SweepFlag,
    TrainFlag,
)
from pytorch_lightning_uv.config import Config, ConfigManager
from pytorch_lightning_uv.data import create_data_module
from pytorch_lightning_uv.data.transform import train_transform
from pytorch_lightning_uv.eval import EDA
from pytorch_lightning_uv.eval.logger import LoggerManager
from pytorch_lightning_uv.model import MLP
from pytorch_lightning_uv.utils import create_rich_progress_bar, set_random_seed

set_random_seed()
torch.set_float32_matmul_precision("high")


def training(config: Config) -> None:
    # logger
    with LoggerManager(
        run_name=config.logger.run_name,
        entity=config.logger.entity,
        project=config.logger.project,
        log_model=True,  # enable log model ckpt
        config=config,
    ) as logger:
        # dataset
        datamodule = create_data_module(
            name=config.data.dataset,
            batch_size=config.data.batch_size,
            transforms=train_transform(),
        )
        datamodule.prepare_data()
        datamodule.setup("fit")
        # model
        model = MLP(
            n_layer_1=config.model.n_layer_1,
            n_layer_2=config.model.n_layer_2,
            lr=config.optimizer.lr,
            dropout_rate=config.model.dropout,
        )
        # log model ckpt and gradients
        if not logger.sweeping:
            logger.watch(model, log="all")
            logger.upload_best_model()

        # trainer
        trainer = Trainer(
            logger=logger,
            callbacks=[create_rich_progress_bar()],
            accelerator="gpu",
            max_epochs=config.training.max_epochs,
        )
        trainer.fit(model, datamodule)


def evaluation(config: Config, run_id: str) -> None:
    """Test the model from a specific wandb run.

    Args:
        config_path: Path to config file
        run_id: The wandb run ID to test (printed at end of training)
    """
    # data
    datamodule = create_data_module(
        name=config.data.dataset,
        batch_size=config.data.batch_size,
        transforms=train_transform(),
    )
    datamodule.prepare_data()
    datamodule.setup("test")

    with LoggerManager(
        run_name=config.logger.run_name,
        entity=config.logger.entity,
        project=config.logger.project,
        id=run_id,
        config=config,
        job_type="eval",
    ) as logger:
        # model
        model_path = logger.load_best_model(run_id)
        model = MLP.load_from_checkpoint(model_path)
        model.eval()
        # trainer
        trainer = Trainer(
            logger=logger, accelerator="gpu", callbacks=[create_rich_progress_bar()]
        )
        trainer.test(model, datamodule)


def main(
    config_file: ConfigPath = Path("data/train.yml"),
    eda: EDAFlag = False,
    train: TrainFlag = False,
    eval: EvalFlag = False,
    sweep: SweepFlag = False,
    sweep_config: SweepConfigPath = None,
    sweep_count: SweepCount = 10,
    run_id: RunID = None,
) -> None:
    """
    ML Training and Evaluation CLI:

    - Exploratory Data Analysis (EDA)\n
    - Model Training\n
    - Model Evaluation\n
    - Hyperparameter Sweeps
    """
    config_manager = ConfigManager()
    config = config_manager.load_config(config_file)

    if eda:
        EDA.analyze_dataset(config)
    elif train:
        training(config)
    elif eval and run_id:
        evaluation(config, run_id)
    elif sweep and sweep_config:
        sweep_id = LoggerManager.init_sweep(
            sweep_config_path=sweep_config,
            project=config.logger.project,
            entity=config.logger.entity,
        )
        wandb.agent(
            sweep_id,
            entity=config.logger.entity,
            project=config.logger.project,
            function=lambda: training(config),
            count=sweep_count,
        )
    else:
        options = {
            "config_file": config_file,
            "eda": eda,
            "train": train,
            "eval": eval,
            "run_id": run_id,
            "sweep": sweep,
            "sweep_config": sweep_config,
            "sweep_count": sweep_count,
        }
        raise ValueError(
            "Invalid combination of options:\n"
            + "\n".join(f"{k}={v}" for k, v in options.items())
        )


if __name__ == "__main__":
    typer.run(main)
