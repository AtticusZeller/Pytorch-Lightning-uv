from dataclasses import asdict

import wandb

from pytorch_lightning_uv.config import Config
from pytorch_lightning_uv.eval.logger import LoggerManager


def test_wandb_logger_init(train_config: Config, cleanup_wandb: None) -> None:
    """Test WandbLogger initialization"""

    run_name = "test_run"
    entity = train_config.logger.entity
    project = train_config.logger.project
    test_config = asdict(train_config)
    test_config.pop("logger")

    # Initialize logger with config
    logger_manager = LoggerManager(
        run_name=run_name, entity=entity, project=project, config=test_config
    )

    # Verify wandb run is initialized correctly
    assert logger_manager.logger.experiment is not None
    assert logger_manager.logger.experiment.name == run_name
    assert logger_manager.logger.experiment.project == project
    assert logger_manager.logger.experiment.entity == entity
    assert (
        logger_manager.logger.experiment.config["optimizer"]["lr"]
        == test_config["optimizer"]["lr"]
    )
    # clean
    wandb.finish()
