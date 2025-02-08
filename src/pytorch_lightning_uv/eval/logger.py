from pathlib import Path
from typing import Self

import wandb
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities.types import _EVALUATE_OUTPUT
from rich.pretty import pprint


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
        config: dict | None = None,
    ) -> None:
        config = config or {}
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
        pprint(config)

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
        if self.job_type == "train":
            print("\nTraining completed! To test this model, use:")
            print(f"Run ID: {self.version}")

    def log_test_results(self, metrics: _EVALUATE_OUTPUT) -> None:
        """Log test results to wandb."""
        columns = ["Metric", "Value"]
        data = [[k, v] for k, v in metrics[0].items()]
        self.log_text(key="test_results", columns=columns, data=data)

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

    def watch(self, model, log="all", log_freq=100, log_graph=True) -> None:
        """Override watch method to keep track of watched models."""
        super().watch(model, log=log, log_freq=log_freq, log_graph=log_graph)
        self._watched_models.append(model)
