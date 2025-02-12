from pathlib import Path

import lightning.pytorch as pl
import torch
from torch import Tensor
from torch.nn import BatchNorm1d, CrossEntropyLoss, Dropout, Linear, functional as F
from torch.optim import Adam
from torchmetrics.functional import accuracy

from ailab.config import Config


class BaseModel(pl.LightningModule):
    """MINST MLP model
    Ref: https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch-lightning/Optimize_Pytorch_Lightning_models_with_Weights_%26_Biases.ipynb#scrollTo=gzaiGUAz1saI
    """

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """needs to return a loss from a single batch"""
        _, loss, acc = self._get_preds_loss_accuracy(batch)

        # Log loss and metric
        self.log("train_loss", loss)
        self.log("train_accuracy", acc)

        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """used for logging metrics"""
        preds, loss, acc = self._get_preds_loss_accuracy(batch)

        # Log loss and metric
        self.log("val_loss", loss)
        self.log("val_accuracy", acc)

        # Let's return preds to use it in a custom callback
        return preds

    def test_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> None:
        """used for logging metrics"""
        _, loss, acc = self._get_preds_loss_accuracy(batch)

        # Log loss and metric
        self.log("test_loss", loss)
        self.log("test_accuracy", acc)

    def configure_optimizers(self) -> Adam:
        """defines model optimizer"""
        return Adam(self.parameters(), lr=self.lr)

    def _get_preds_loss_accuracy(
        self, batch: tuple[Tensor, Tensor]
    ) -> tuple[Tensor, Tensor, Tensor]:
        """convenience function since train/valid/test steps are similar"""
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        loss = self.loss(logits, y)
        acc = accuracy(preds, y, "multiclass", num_classes=10)
        return preds, loss, acc


class MLP(BaseModel):
    """MINST MLP model
    Ref: https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch-lightning/Optimize_Pytorch_Lightning_models_with_Weights_%26_Biases.ipynb#scrollTo=gzaiGUAz1saI
    """

    def __init__(
        self,
        n_classes: int = 10,
        n_layer_1: int = 128,
        n_layer_2: int = 256,
        lr: float = 1e-3,
        dropout_rate: float = 0.2,
    ) -> None:
        super().__init__()

        # mnist images are (1, 28, 28) (channels, width, height)
        self.layer_1 = Linear(28 * 28, n_layer_1)
        self.bn_1 = BatchNorm1d(n_layer_1)
        self.dropout_1 = Dropout(dropout_rate)
        self.layer_2 = Linear(n_layer_1, n_layer_2)
        self.bn_2 = BatchNorm1d(n_layer_2)
        self.dropout_2 = Dropout(dropout_rate)
        self.layer_3 = Linear(n_layer_2, n_classes)

        # loss
        self.loss = CrossEntropyLoss()

        # optimizer parameters
        self.lr = lr

        # save hyper-parameters to self.hparams (auto-logged by W&B)
        self.save_hyperparameters()

    def forward(self, x: Tensor) -> Tensor:
        batch_size, channels, width, height = x.size()

        # (b, 1, 28, 28) -> (b, 1*28*28)
        x = x.view(batch_size, -1)

        # let's do 3 x (linear + relu)
        x = self.layer_1(x)
        x = self.bn_1(x)
        x = F.relu(x)
        x = self.dropout_1(x)

        x = self.layer_2(x)
        x = self.bn_2(x)
        x = F.relu(x)
        x = self.dropout_2(x)

        x = self.layer_3(x)

        return x


def create_model(config: Config, model_path: Path | None = None) -> BaseModel:
    if config.model.name.lower() == "mlp":
        return (
            MLP(
                n_layer_1=config.model.n_layer_1,
                n_layer_2=config.model.n_layer_2,
                lr=config.optimizer.lr,
                dropout_rate=config.model.dropout,
            )
            if model_path is None
            else MLP.load_from_checkpoint(model_path)
        )
    else:
        raise ValueError(f"Model name {config.model.name} not supported.")
