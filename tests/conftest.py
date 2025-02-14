from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest

import wandb
from ailab.config import Config, ConfigManager
from ailab.utils import set_random_seed


@pytest.fixture(scope="session")
def set_seed() -> Generator[None, Any, None]:
    set_random_seed()
    yield


@pytest.fixture(scope="function")
def config_path(tmp_path: Path) -> Path:
    return tmp_path / "config"


@pytest.fixture
def config_manager(config_path: Path) -> ConfigManager:
    return ConfigManager(config_path)


@pytest.fixture
def train_config(config_manager: ConfigManager, config_path: Path) -> Config:
    config_manager.generate_default_configs()
    return config_manager.load_config(config_path / "train.yml")


@pytest.fixture
def cleanup_wandb(train_config: Config) -> Generator[None, Any, None]:
    yield
    # Delete test runs after all tests
    api = wandb.Api()
    runs = api.runs(f"{train_config.logger.entity}/{train_config.logger.project}")
    for run in runs:
        if run.name.startswith("test_"):
            run.delete()
