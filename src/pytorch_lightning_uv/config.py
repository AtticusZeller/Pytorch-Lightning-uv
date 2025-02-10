# config.py
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml
from rich import print


@dataclass
class ModelConfig:
    name: str = "MLP"
    n_layer_1: int = 128
    n_layer_2: int = 256
    dropout: float | None = None
    activation: str = "relu"


@dataclass
class OptimizerConfig:
    name: str = "adam"
    lr: float = 1e-4


@dataclass
class DataConfig:
    dataset: str = "MNIST"
    batch_size: int = 128
    augmentation: list[str] | None = None


@dataclass
class TrainingConfig:
    max_epochs: int = 25
    gradient_clip_val: float | None = None
    accumulate_grad_batches: int | None = None
    precision: int | None = None


@dataclass
class LoggerConfig:
    run_name: str = "test_run"
    entity: str = "atticux"  # set to name of your wandb team
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
        # sub config
        for name, component in self.config_map.items():
            component_dir = self.config_dir / name
            component_dir.mkdir(exist_ok=True)

            conf = asdict(component())
            config_path = component_dir / "default.yml"
            self._save_config({name: conf}, config_path)

        # basic train config
        base_config = {
            "model": "model/default.yml",
            "optimizer": "optimizer/default.yml",
            "data": "data/default.yml",
            "training": "training/default.yml",
            "logger": "logger/default.yml",
        }
        self._save_config(base_config, self.config_dir / "train.yml")

    def load_config(self, config_path: str | Path) -> Config:
        """load configuration from base config"""
        config_path = Path(config_path)
        if not config_path.exists() or not config_path.is_file():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        print(f"Loading config from: [bold cyan]{config_path}[/bold cyan]")
        conf = self._load_config(config_path)
        if config_path.parent == self.config_dir:
            # load all config
            for name, component in conf.items():
                # filter something like `model: model/default.yml`
                if name in self.config_map and isinstance(component, str):
                    config_dict = self._load_config(self.config_dir.joinpath(component))
                    # replace str with dataclass
                    conf[name] = self.config_map[name](**config_dict[name])

            return Config(**conf)
        else:
            raise ValueError(f"Invalid config file: {config_path}")

    @staticmethod
    def _save_config(config: dict[str, str], save_path: Path) -> None:
        with open(save_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

    @staticmethod
    def _load_config(config_path: Path) -> dict:
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
