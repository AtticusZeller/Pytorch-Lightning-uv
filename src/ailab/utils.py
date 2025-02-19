from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from torch.utils.data import DataLoader

from ailab.data.dataset import DataModule


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


def mean_std(data_module: DataModule) -> tuple[float, float]:
    dataset = data_module.data(
        data_module.data_dir, train=True, transform=data_module.transform
    )
    loader = DataLoader(dataset, batch_size=len(dataset), pin_memory=True)

    data = next(iter(loader))[0]
    return data.mean().item(), data.std().item()
