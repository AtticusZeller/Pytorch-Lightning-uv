from pathlib import Path

import pytest

from pytorch_lightning_uv.config import Config, ConfigManager


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
def eval_config(config_manager: ConfigManager, config_path: Path) -> Config:
    config_manager.generate_default_configs()
    return config_manager.load_config(config_path / "eval.yml")
