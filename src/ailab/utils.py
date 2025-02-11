import random

import numpy as np
import torch
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme


def set_random_seed(seed: int = 42) -> None:
    """Ensure deterministic behavior."""
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def create_rich_progress_bar() -> RichProgressBar:
    """Create a RichProgressBar instance.
    Ref: https://lightning.ai/docs/pytorch/stable/common/progress_bar.html"""
    return RichProgressBar(
        theme=RichProgressBarTheme(
            description="green_yellow",
            progress_bar="green1",
            progress_bar_finished="green1",
            progress_bar_pulse="#6206E0",
            batch_progress="green_yellow",
            time="grey82",
            processing_speed="grey82",
            metrics="grey82",
            metrics_text_delimiter="\n",
            metrics_format=".3e",
        )
    )
