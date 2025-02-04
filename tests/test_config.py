from pathlib import Path

import pytest

from pytorch_lightning_uv.config import (
    Config,
    ConfigManager,
    DataConfig,
    LoggerConfig,
    ModelConfig,
    OptimizerConfig,
    TrainingConfig,
)


@pytest.fixture(scope="function")
def config_path(tmp_path: Path) -> Path:
    return tmp_path / "config"


@pytest.fixture
def config_manager(config_path: Path):
    return ConfigManager(config_dir=config_path)


def test_generate_default_configs(config_manager, config_path):
    config_manager.generate_default_configs()

    # Check main config files exist
    assert (config_path / "train.yml").exists()
    assert (config_path / "eval.yml").exists()

    # Check component directories and configs exist
    for component in ["model", "optimizer", "data", "training", "logger"]:
        assert (config_path / component).is_dir()
        assert (config_path / component / "default.yml").exists()


def test_load_full_train_config(config_manager, config_path):
    config_manager.generate_default_configs()
    config = config_manager.load_config(config_path / "train.yml")

    assert isinstance(config, Config)
    assert isinstance(config.model, ModelConfig)
    assert isinstance(config.optimizer, OptimizerConfig)
    assert isinstance(config.data, DataConfig)
    assert isinstance(config.training, TrainingConfig)
    assert isinstance(config.logger, LoggerConfig)
    assert config.seed == 42


def test_load_eval_config(config_manager, config_path):
    config_manager.generate_default_configs()
    config = config_manager.load_config(config_path / "eval.yml")

    assert isinstance(config, Config)
    assert isinstance(config.model, ModelConfig)
    assert isinstance(config.data, DataConfig)
    assert isinstance(config.logger, LoggerConfig)
    assert config.optimizer is None
    assert config.training is None
    assert config.seed == 42


def test_load_component_config(config_manager, config_path):
    config_manager.generate_default_configs()
    model_config = config_manager.load_config(config_path / "model/default.yml")

    assert isinstance(model_config, ModelConfig)
    assert model_config.name == "resnet18"
    assert model_config.hidden_size == 512


def test_load_nonexistent_config(config_manager):
    with pytest.raises(FileNotFoundError):
        config_manager.load_config("nonexistent.yml")
