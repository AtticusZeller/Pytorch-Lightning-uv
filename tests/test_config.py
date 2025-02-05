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


def test_generate_default_configs(config_manager: ConfigManager, config_path: Path):
    config_manager.generate_default_configs()

    # Check main config files exist
    assert (config_path / "train.yml").exists()
    assert (config_path / "eval.yml").exists()

    # Check component directories and configs exist
    for component in ["model", "optimizer", "data", "training", "logger"]:
        assert (config_path / component).is_dir()
        assert (config_path / component / "default.yml").exists()


def test_load_full_train_config(config_manager: ConfigManager, config_path: Path):
    config_manager.generate_default_configs()
    config = config_manager.load_config(config_path / "train.yml")

    assert isinstance(config, Config)
    assert isinstance(config.model, ModelConfig)
    assert isinstance(config.optimizer, OptimizerConfig)
    assert isinstance(config.data, DataConfig)
    assert isinstance(config.training, TrainingConfig)
    assert isinstance(config.logger, LoggerConfig)
    assert config.seed == 42


def test_load_eval_config(config_manager: ConfigManager, config_path: Path):
    config_manager.generate_default_configs()
    config = config_manager.load_config(config_path / "eval.yml")

    assert isinstance(config, Config)
    assert isinstance(config.model, ModelConfig)
    assert isinstance(config.data, DataConfig)
    assert isinstance(config.logger, LoggerConfig)
    assert config.optimizer is None
    assert config.training is None
    assert config.seed == 42


def test_load_component_config(config_manager: ConfigManager, config_path: Path):
    config_manager.generate_default_configs()
    model_config = config_manager.load_config(config_path / "model/default.yml")

    assert isinstance(model_config, ModelConfig)
    assert model_config.name == "resnet18"
    assert model_config.hidden_size == 512


def test_load_nonexistent_config(config_manager: ConfigManager):
    with pytest.raises(FileNotFoundError):
        config_manager.load_config("nonexistent.yml")


def test_config_as_dict(config_manager: ConfigManager, config_path: Path):
    config_manager.generate_default_configs()
    config = config_manager.load_config(config_path / "train.yml")
    config_dict = config.as_dict()

    assert isinstance(config_dict, dict)
    assert "model" in config_dict
    assert "logger" in config_dict
    assert "data" in config_dict
    assert "training" in config_dict
    assert "optimizer" in config_dict
    assert "seed" in config_dict
    assert config_dict["seed"] == 42

    # Verify dict contents match config object
    assert config_dict["model"]["name"] == config.model.name
    assert config_dict["optimizer"]["lr"] == config.optimizer.lr
    assert config_dict["data"]["dataset"] == config.data.dataset
    assert config_dict["training"]["max_epochs"] == config.training.max_epochs
