from dataclasses import asdict
from pathlib import Path

import pytest
import yaml

from expt.config import (
    Config,
    ConfigManager,
    DataConfig,
    LoggerConfig,
    ModelConfig,
    OptimizerConfig,
    TrainingConfig,
)


def test_generate_default_configs(config_manager: ConfigManager, config_path: Path):
    config_manager.generate_default_configs()
    assert (config_path / "train.yml").exists()

    # Verify the content of generated config
    with open(config_path / "train.yml") as f:
        config = yaml.safe_load(f)

    assert all(
        key in config for key in ["model", "optimizer", "data", "training", "logger"]
    )
    assert config["model"]["name"] == "MLP"
    assert config["optimizer"]["lr"] == 1e-4
    assert config["data"]["dataset"] == "MNIST"


def test_load_config(config_manager: ConfigManager, config_path: Path):
    config_manager.generate_default_configs()
    config = config_manager.load_config(config_path / "train.yml")

    assert isinstance(config, Config)
    assert isinstance(config.model, ModelConfig)
    assert isinstance(config.optimizer, OptimizerConfig)
    assert isinstance(config.data, DataConfig)
    assert isinstance(config.training, TrainingConfig)
    assert isinstance(config.logger, LoggerConfig)

    # Test default values
    assert config.model.name == "MLP"
    assert config.optimizer.lr == 1e-4
    assert config.data.batch_size == 128
    assert config.training.max_epochs == 25
    assert config.logger.project == "pytorch-lightning-uv"


def test_load_nonexistent_config(config_manager: ConfigManager, config_path: Path):
    with pytest.raises(FileNotFoundError):
        config_manager.load_config("nonexistent.yml")

    with pytest.raises(FileNotFoundError):
        config_manager.load_config(config_path / "train.json")

    with pytest.raises(FileNotFoundError):
        config_manager.load_config(config_path)


def test_config_as_dict(config_manager: ConfigManager, config_path: Path):
    config_manager.generate_default_configs()
    config = config_manager.load_config(config_path / "train.yml")
    config_dict = asdict(config)

    assert isinstance(config_dict, dict)
    assert all(
        key in config_dict
        for key in ["model", "optimizer", "data", "training", "logger"]
    )

    # Verify dict contents match config object
    assert config_dict["model"]["name"] == config.model.name
    assert config_dict["optimizer"]["lr"] == config.optimizer.lr
    assert config_dict["data"]["dataset"] == config.data.dataset
    assert config_dict["training"]["max_epochs"] == config.training.max_epochs
