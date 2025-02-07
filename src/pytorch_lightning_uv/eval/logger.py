from pathlib import Path

import wandb
from lightning.pytorch.loggers import WandbLogger


class LoggerManager:
    """
    Initialize the Weights & Biases logging.
    ```bash
    wandb login --relogin --host=https://wandb.atticux.me
    ```
    Ref:
        1. https://docs.wandb.ai/guides/integrations/lightning/
        2. https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.wandb.html#module-lightning.pytorch.loggers.wandb
    """

    def __init__(
        self,
        run_name: str,
        entity: str,
        project: str,
        config: dict | None = None,
        base_url: str = "https://wandb.atticux.me",
    ) -> None:
        config = config or {}
        Path("./logs").mkdir(parents=True, exist_ok=True)
        self.logger = WandbLogger(
            project=project,
            entity=entity,
            name=run_name,
            config=config,
            save_dir="./logs",
            log_model="all",
            # offline=True,
            settings=wandb.Settings(base_url=base_url),
        )

        print(f"Run name: {run_name}:config: {config}")

    # TODO： add checkpoint logger
    # TODO: logger image
