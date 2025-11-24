from pathlib import Path
from typing import Literal

import lightning.pytorch as pl
import timm
import torch
from rich import print
from torch import Tensor, nn
from torch.nn import BatchNorm1d, CrossEntropyLoss, Dropout, Linear, functional as F
from torch.optim import Adam, Optimizer
from torchmetrics.functional import accuracy

from expt.config import Config
from expt.utils import check_transform


class BaseModel(pl.LightningModule):
    """MINST MLP model
    Ref: https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch-lightning/Optimize_Pytorch_Lightning_models_with_Weights_%26_Biases.ipynb#scrollTo=gzaiGUAz1saI
    """

    loss: CrossEntropyLoss
    lr: float

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

    def configure_optimizers(self) -> Optimizer:
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
        num_classes: int = 10,
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
        self.layer_3 = Linear(n_layer_2, num_classes)

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


class CNN(BaseModel):
    """CNN model for Fashion MNIST dataset
    Architecture:
    - 2 Convolutional layers followed by max pooling
    - 2 Fully connected layers
    - Batch normalization and dropout for regularization
    """

    def __init__(
        self,
        num_classes: int = 10,
        n_channels_1: int = 32,
        n_channels_2: int = 64,
        n_fc_1: int = 128,
        lr: float = 1e-3,
        dropout_rate: float = 0.2,
    ) -> None:
        super().__init__()

        # First conv block
        self.conv1 = torch.nn.Conv2d(1, n_channels_1, kernel_size=5, padding=2)
        self.bn1 = torch.nn.BatchNorm2d(n_channels_1)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2)
        self.dropout1 = Dropout(dropout_rate)

        # Second conv block
        self.conv2 = torch.nn.Conv2d(
            n_channels_1, n_channels_2, kernel_size=3, padding=1
        )
        self.bn2 = torch.nn.BatchNorm2d(n_channels_2)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2)
        self.dropout2 = Dropout(dropout_rate)

        # Calculate size after convolutions and pooling
        # Input: 28x28 -> Conv1: 28x28 -> Pool1: 14x14 -> Conv2: 14x14 -> Pool2: 7x7
        conv_output_size = 7 * 7 * n_channels_2

        # Fully connected layers
        self.fc1 = Linear(conv_output_size, n_fc_1)
        self.bn3 = BatchNorm1d(n_fc_1)
        self.dropout3 = Dropout(dropout_rate)
        self.fc2 = Linear(n_fc_1, num_classes)

        # loss
        self.loss = CrossEntropyLoss()

        # optimizer parameters
        self.lr = lr

        # save hyper-parameters to self.hparams (auto-logged by W&B)
        self.save_hyperparameters()

    def forward(self, x: Tensor) -> Tensor:
        # First conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.fc1(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout3(x)
        x = self.fc2(x)

        return x


class FineTuneBaseModel(BaseModel):
    model: nn.Module

    def freeze_except(self, trainable: list[str], debug: bool = False) -> None:
        """freeze all layers except the ones specified in trainable

        Parameters
        ----------
        trainable : list[str]
            list of layer names to be unfrozen
        debug : bool, optional
            print all finetune layers, by default False
        """
        # Freeze all layers
        for _, param in self.model.named_parameters():
            param.requires_grad = False
        # Unfreeze the layers
        fine_tune_params = []
        fine_tune_layers = set()
        for module_name, module in self.model.named_modules():
            if module_name in trainable:
                fine_tune_layers.add(module_name)
                for name, param in module.named_parameters():
                    param.requires_grad = True
                    fine_tune_params.append(module_name + "." + name)
        print(f"Fine-tuning {len(fine_tune_layers)} layers:")
        layers = list(fine_tune_layers)
        layers.sort()
        print(layers)
        if debug:
            print(f"Fine-tuning {len(fine_tune_params)} paras:")
            print(fine_tune_params)


class ResNet18Transfer(FineTuneBaseModel):
    """ResNet18 transfer learning model for Fashion MNIST
    Features:
    - Uses pretrained ResNet18 as backbone
    - Custom classification head
    - Supports feature extraction and fine-tuning
    """

    def __init__(
        self,
        num_classes: int = 10,
        lr: float = 1e-3,
        unfreeze_layers: list[str] | None = None,
    ) -> None:
        super().__init__()

        # Load ResNet18 model without pretrained weights
        self.model = timm.create_model(
            "resnet18", pretrained=True, num_classes=num_classes
        )
        # check_transform(self.model)
        # loss
        self.lr = lr
        self.loss = nn.CrossEntropyLoss()

        # save hyperparameters
        self.save_hyperparameters()
        if unfreeze_layers is not None:
            self.freeze_except(unfreeze_layers)
        # self.resnet = torch.compile(self.resnet)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class EfficientNetV2Transfer(FineTuneBaseModel):
    """EfficientNet transfer learning model for Fashion MNIST
    Features:
    - Uses pretrained EfficientNet as backbone
    - Custom classification head
    - Supports feature extraction and fine-tuning
    Ref: https://lightning.ai/docs/pytorch/stable/advanced/transfer_learning.html#example-imagenet-computer-vision
    """

    def __init__(
        self,
        num_classes: int = 10,
        efficient_version: Literal["s", "m", "l"] = "s",
        lr: float = 1e-3,
        unfreeze_layers: list[str] | None = None,
    ) -> None:
        super().__init__()

        self.model = timm.create_model(
            f"tf_efficientnetv2_{efficient_version}",
            pretrained=True,
            num_classes=num_classes,
        )
        check_transform(self.model)
        # loss
        self.lr = lr
        self.loss = nn.CrossEntropyLoss()

        # save hyperparameters
        self.save_hyperparameters()
        if unfreeze_layers is not None:
            self.freeze_except(unfreeze_layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


def create_model(config: Config, model_path: Path | None = None) -> BaseModel:
    if config.model.name.lower() == "mlp":
        return (
            MLP(
                n_layer_1=config.model.n_layer_1,  # type: ignore
                n_layer_2=config.model.n_layer_2,  # type: ignore
                lr=config.optimizer.lr,
                dropout_rate=config.model.dropout,
            )
            if model_path is None
            else MLP.load_from_checkpoint(model_path)
        )
    elif config.model.name.lower() == "cnn":
        return (
            CNN(
                n_channels_1=config.model.n_channels_1,  # type: ignore
                n_channels_2=config.model.n_channels_2,  # type: ignore
                n_fc_1=config.model.n_fc_1,  # type: ignore
                lr=config.optimizer.lr,
                dropout_rate=config.model.dropout,
            )
            if model_path is None
            else CNN.load_from_checkpoint(model_path)
        )
    elif config.model.name.lower() == "resnet18":
        return (
            ResNet18Transfer(
                lr=config.optimizer.lr, unfreeze_layers=config.model.unfreeze_layers
            )
            if model_path is None
            else ResNet18Transfer.load_from_checkpoint(model_path)
        )
    elif config.model.name.lower() == "efficientnet_v2":
        return (
            EfficientNetV2Transfer(
                lr=config.optimizer.lr,
                efficient_version=config.model.efficient_version or "s",
                unfreeze_layers=config.model.unfreeze_layers,
            )
            if model_path is None
            else EfficientNetV2Transfer.load_from_checkpoint(model_path)
        )

    else:
        raise ValueError(f"Model name {config.model.name} not supported.")
