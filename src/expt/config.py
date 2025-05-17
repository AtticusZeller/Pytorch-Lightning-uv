# config.py
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from rich import print


@dataclass
class ModelConfig:
    name: str = "MLP"
    dropout: float = 0.2
    activation: str | None = None
    # MLP
    n_layer_1: int | None = None
    n_layer_2: int | None = None
    # CNN
    n_channels_1: int | None = None
    n_channels_2: int | None = None
    n_fc_1: int | None = None
    # EfficientNet
    efficient_version: Literal["s", "m", "l"] | None = None
    # Fine-tuning
    unfreeze_layers: list[str] | None = None


@dataclass
class OptimizerConfig:
    name: str = "adam"
    lr: float = 1e-4


@dataclass
class DataConfig:
    dataset: str = "MNIST"
    batch_size: int = 128
    augmentation: list[str] | None = None
    transform: Literal["standardize", "base"] = "standardize"


@dataclass
class TrainingConfig:
    max_epochs: int = 25
    gradient_clip_val: float | None = None
    accumulate_grad_batches: int | None = None
    precision: int | None = None


@dataclass
class LoggerConfig:
    run_name: str = "test_run"
    entity: str = "your_wandb_entity"  # set to name of your wandb team
    project: str = "pytorch-lightning-uv"


@dataclass
class Config:
    model: ModelConfig
    logger: LoggerConfig
    data: DataConfig
    training: TrainingConfig
    optimizer: OptimizerConfig


class ConfigManager:
    def __init__(self, config_dir: str | Path = "./config") -> None:
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.config_map = {
            "model": ModelConfig,
            "optimizer": OptimizerConfig,
            "data": DataConfig,
            "training": TrainingConfig,
            "logger": LoggerConfig,
        }

    def generate_default_configs(self) -> None:
        """generate default configuration files for evaluation and training"""
        print("Generating default configuration files...")
        default_config = {}
        # sub config
        for name, component in self.config_map.items():
            sub_conf = asdict(component())
            default_config[name] = sub_conf
        self._save_config(default_config, self.config_dir / "train.yml")

    def load_config(self, config_path: str | Path) -> Config:
        """load configuration from yml file"""
        config_path = Path(config_path)
        if (
            not config_path.exists()
            or not config_path.is_file()
            or config_path.suffix not in [".yml", ".yaml"]
        ):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        print(f"Loading config from: [bold cyan]{config_path}[/bold cyan]")
        conf = self._load_config(config_path)
        # load all config
        for name in conf:
            if name in self.config_map:
                conf[name] = self.config_map[name](**conf[name])

        return Config(**conf)

    @staticmethod
    def _save_config(config: dict[str, Any], save_path: Path) -> None:
        with open(save_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

    @staticmethod
    def _load_config(config_path: Path) -> dict[str, Any]:
        with open(config_path) as f:
            return yaml.safe_load(f)


if __name__ == "__main__":
    config_manager = ConfigManager()
    config_manager.generate_default_configs()
    config = config_manager.load_config("config/train.yml")

    print("\nConfiguration loaded successfully:")
    print(f"Model name: {config.model.name}")
    print(f"Learning rate: {config.optimizer.lr}")
    print(f"Dataset: {config.data.dataset}")
    print(f"Max epochs: {config.training.max_epochs}")
