from dataclasses import asdict
from pathlib import Path
from typing import Self

import yaml
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from rich import print
from rich.pretty import pprint

import wandb
from pytorch_lightning_uv.config import (
    Config,
    DataConfig,
    ModelConfig,
    OptimizerConfig,
    TrainingConfig,
)


class LoggerManager(WandbLogger):
    """
    Initialize the Weights & Biases logging.
    ```bash
    wandb login --relogin --host=https://wandb.atticux.me
    ```
    Ref:
        1. https://docs.wandb.ai/ref/python/init/
        2. https://docs.wandb.ai/guides/integrations/lightning/
        3. https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.wandb.html#module-lightning.pytorch.loggers.wandb
    """

    def __init__(
        self,
        run_name: str,
        entity: str,
        project: str,
        id: str | None = None,
        log_model: bool = True,
        job_type: str = "train",
        base_url: str = "https://wandb.atticux.me",
        config: Config | None = None,
    ) -> None:
        Path("./logs").mkdir(parents=True, exist_ok=True)
        super().__init__(
            project=project,
            entity=entity,
            name=run_name,
            job_type=job_type,
            id=id,
            resume="must"
            if id is not None
            else "never",  # resume run if id is provided
            config=config,
            save_dir="./logs",
            log_model=log_model,  # log model artifacts while Trainer callbacks
            # offline=True,
            settings=wandb.Settings(base_url=base_url),
        )
        self.entity = entity
        self.job_type = job_type

        if config is not None:
            self.config = config

            if self.sweeping:
                # update from sweep
                self._update_config_with_sweep(config)

        pprint(asdict(config))

        self._watched_models = []

    def __enter__(self) -> Self:
        """Enter the context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the context manager.

        Will automatically unwatch all watched models and finish the wandb run.
        """
        # Unwatch all models that were watched
        for model in self._watched_models:
            self.experiment.unwatch(model)

        # Finish the wandb run
        self.experiment.finish()
        if self.job_type == "train" and not self.sweeping:
            print("\nTraining completed! To test this model, use:")
            print(f"Run ID: [bold cyan]{self.version}[/bold cyan]")

    def load_best_model(self, run_id: str) -> Path:
        """Load the best model from a specific run ID."""
        model_dir = self.download_artifact(
            artifact=f"{self.entity}/{self.name}/model-{run_id}:best",
            artifact_type="model",
            use_artifact=True,  # link current run to the artifact
            save_dir=f"./artifacts/{run_id}",
        )
        model_path = Path(model_dir) / "model.ckpt"
        if not model_path.exists():
            raise FileNotFoundError(f"Model-{run_id} not found.")
        return model_path

    def upload_best_model(
        self, monitor: str = "val_accuracy", mode: str = "max"
    ) -> None:
        """Upload the best model to wandb"""
        checkpoint_callback = ModelCheckpoint(
            monitor="val_accuracy",
            mode="max",
            dirpath=f"checkpoints/{self.config.model.name}",
            filename=f"{self.config.model.name}-{self.config.data.dataset}"
            + "-{epoch:02d}-{val_accuracy:.2f}",
            save_top_k=1,
        )
        self.after_save_checkpoint(checkpoint_callback)

    def watch(self, model, log="all", log_freq=100, log_graph=False) -> None:
        """Override watch method to keep track of watched models."""
        super().watch(model, log=log, log_freq=log_freq, log_graph=log_graph)
        self._watched_models.append(model)

    @staticmethod
    def init_sweep(sweep_config_path: Path, project: str, entity: str) -> str:
        """Initialize sweep from config file"""
        with open(sweep_config_path) as f:
            sweep_config = yaml.safe_load(f)

        sweep_id = wandb.sweep(sweep_config, project=project, entity=entity)
        return sweep_id

    @property
    def sweeping(self) -> bool:
        """Check if the current run is part of a sweep"""
        return self.experiment.sweep_id is not None

    def _update_config_with_sweep(self, config: Config) -> Config:
        """Update config passing to experiment and wandb config with sweep values
        if in sweep mode"""
        # derive sweep config from experiment config
        sweep_config = self.experiment.config.as_dict()

        for field in asdict(config):
            sub_config: ModelConfig | OptimizerConfig | DataConfig | TrainingConfig = (
                getattr(config, field)
            )
            for key in asdict(sub_config):
                if key in sweep_config:
                    # update config with sweep values
                    setattr(sub_config, key, sweep_config[key])
                    # update wandb config with sweep values
                    sweep_config[field][key] = sweep_config[key]
        # force update sweep config
        self.experiment.config.update(sweep_config, True)
        return config
