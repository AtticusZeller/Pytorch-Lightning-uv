"""
Ref:
    1. https://lightning.ai/docs/pytorch/stable/debug/debugging_basic.html
    2. https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch-lightning/Optimize_Pytorch_Lightning_models_with_Weights_%26_Biases.ipynb#scrollTo=h__ic9lC1saP
    3. https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelCheckpoint.html#lightning.pytorch.callbacks.ModelCheckpoint
"""

from pathlib import Path

import torch
import typer
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import RichModelSummary
from rich.console import Console

import wandb
from expt.cli import (
    ConfigPath,
    EDAFlag,
    EvalFlag,
    SweepConfigPath,
    SweepCount,
    SweepFlag,
    TrainFlag,
)
from expt.config import Config, ConfigManager
from expt.data import create_data_module
from expt.eval import EDA
from expt.eval.logger import LoggerManager
from expt.model import create_model
from expt.utils import create_rich_progress_bar

seed_everything(42, workers=True)
torch.set_float32_matmul_precision("high")
console = Console()


def training(config: Config) -> str | None:
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
            transform=config.data.transform,
        )
        datamodule.prepare_data()
        datamodule.setup("fit")
        # model
        model = create_model(config)
        # log model ckpt and gradients
        if not logger.sweeping:
            logger.watch(model, log="all")

        # trainer
        trainer = Trainer(
            logger=logger,
            # profiler=PyTorchProfiler(),
            callbacks=[
                RichModelSummary(3),  # print model structure
                create_rich_progress_bar(),
                logger.checkpoint_callback(),
            ],
            accelerator="gpu",
            max_epochs=config.training.max_epochs,
        )
        trainer.fit(model, datamodule)

        run_id = logger.version
    return run_id


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
        transform=config.data.transform,
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
        model = create_model(config, model_path)
        model.eval()
        # trainer
        trainer = Trainer(
            logger=logger,
            accelerator="gpu",
            enable_model_summary=True,
            callbacks=[RichModelSummary(3), create_rich_progress_bar()],
        )
        trainer.test(model, datamodule)


def main(
    config_file: ConfigPath = Path("data/train.yml"),
    eda: EDAFlag = False,
    train: TrainFlag = False,
    eval_id: EvalFlag = None,
    sweep: SweepFlag = False,
    sweep_config: SweepConfigPath = None,
    sweep_count: SweepCount = 10,
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
        try:
            run_id = training(config)
        except Exception:
            console.print_exception(max_frames=1)
        else:
            if run_id:
                evaluation(config, run_id)
    elif eval_id:
        evaluation(config, eval_id)
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
            "eval_id": eval_id,
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
