from dataclasses import asdict

from expt.config import Config
from expt.eval.logger import LoggerManager


def test_wandb_logger_init(train_config: Config, cleanup_wandb: None) -> None:
    """Test WandbLogger initialization"""

    run_name = "test_run"
    train_config.logger.run_name = run_name
    # for test purposes, we set the entity to "atticux"
    train_config.logger.entity = "atticux"
    entity = train_config.logger.entity
    project = train_config.logger.project
    test_config = train_config
    # Initialize logger with config
    with LoggerManager(
        run_name=run_name,
        entity=entity,
        project=project,
        log_model=False,
        job_type="test",
        config=test_config,
    ) as logger_manager:
        # Verify wandb run is initialized correctly
        assert logger_manager.experiment is not None
        assert logger_manager.experiment.name == run_name
        assert logger_manager.experiment.project == project
        assert logger_manager.experiment.entity == entity
        assert (
            logger_manager.experiment.config["optimizer"]["lr"]
            == asdict(test_config)["optimizer"]["lr"]
        )
