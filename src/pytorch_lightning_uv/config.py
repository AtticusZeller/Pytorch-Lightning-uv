# config.py
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from omegaconf import OmegaConf


@dataclass
class ModelConfig:
    name: str = "resnet18"
    hidden_size: int = 512
    num_layers: int = 18
    dropout: float = 0.1
    activation: str = "relu"


@dataclass
class OptimizerConfig:
    name: str = "adam"
    lr: float = 0.001
    weight_decay: float = 0.0001
    betas: tuple = (0.9, 0.999)


@dataclass
class DataConfig:
    dataset: str = "MNIST"
    batch_size: int = 128
    num_workers: int = 4
    augmentation: list[str] = field(
        default_factory=lambda: ["random_crop", "random_flip"]
    )


@dataclass
class TrainingConfig:
    max_epochs: int = 10
    gradient_clip_val: float | None = 1.0
    accumulate_grad_batches: int = 1
    precision: int = 32


@dataclass
class LoggerConfig:
    run_name: str | None = None
    config: dict | None = None
    entity: str = "atticux"  # set to name of your wandb team
    project: str = "pytorch-lightning-uv"


@dataclass
class Config:
    model: ModelConfig
    logger: LoggerConfig
    data: DataConfig
    training: TrainingConfig | None = None
    optimizer: OptimizerConfig | None = None
    seed: int = 42


class ConfigManager:
    def __init__(self, config_dir: str | Path = "./config") -> None:
        self.config_dir = Path(config_dir)
        self.config_map = {
            "model": ModelConfig,
            "optimizer": OptimizerConfig,
            "data": DataConfig,
            "training": TrainingConfig,
            "logger": LoggerConfig,
        }

    def generate_default_configs(self) -> None:
        """generate default configuration files for evaluation and training"""
        self.config_dir.mkdir(parents=True, exist_ok=True)
        # sub config
        for name, component in self.config_map.items():
            component_dir = self.config_dir / name
            component_dir.mkdir(exist_ok=True)

            # dataclass to OmegaConf
            conf = OmegaConf.structured(component())
            config_path = component_dir / "default.yml"
            self._save_config({name: OmegaConf.to_container(conf)}, config_path)

        # basic train config
        base_config = {
            "seed": Config.seed,
            "model": "model/default.yml",
            "optimizer": "optimizer/default.yml",
            "data": "data/default.yml",
            "training": "training/default.yml",
            "logger": "logger/default.yml",
        }
        self._save_config(base_config, self.config_dir / "train.yml")

        # basic eval config
        base_config = {
            "seed": Config.seed,
            "model": "model/default.yml",
            "data": "data/default.yml",
            "logger": "logger/default.yml",
        }
        self._save_config(base_config, self.config_dir / "eval.yml")

    def load_config(
        self, config_path: str | Path
    ) -> Config | ModelConfig | OptimizerConfig | DataConfig | TrainingConfig:
        """load configuration from base config or sub-config"""
        config_path = Path(config_path)
        if not config_path.exists() or not config_path.is_file():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        conf = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
        if config_path.parent == self.config_dir:
            # load all config
            for name, component in conf.items():
                # filter something like `model: model/default.yml`
                if name in self.config_map and isinstance(component, str):
                    config_dict = OmegaConf.load(self.config_dir.joinpath(component))
                    # replace str with dataclass
                    conf[name] = self.config_map[name](**config_dict[name])

            return Config(**conf)
        elif config_path.parent.name in self.config_map:
            # load single config
            config_dict = OmegaConf.load(config_path)
            return self.config_map[config_path.parent.name](
                **config_dict[config_path.parent.name]
            )
        else:
            raise ValueError(f"Invalid config file: {config_path}")

    @staticmethod
    def _save_config(config: dict, save_path: Path) -> None:
        with open(save_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)


if __name__ == "__main__":
    config_manager = ConfigManager()
    print("Generating default configuration files...")
    config_manager.generate_default_configs()

    print("\nLoading configuration...")
    config = config_manager.load_config("config/train.yml")

    print("\nConfiguration loaded successfully:")
    print(f"Model name: {config.model.name}")
    print(f"Learning rate: {config.optimizer.lr}")
    print(f"Dataset: {config.data.dataset}")
    print(f"Max epochs: {config.training.max_epochs}")
