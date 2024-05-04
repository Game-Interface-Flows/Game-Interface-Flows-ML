import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchmetrics import Accuracy, Precision
from torchvision.models import resnet18


class ClassificationModel(pl.LightningModule):
    def __init__(self, input_shape=(3, 256, 256), learning_rate: float = 0.01):
        super().__init__()
        self.model = resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)
        self.loss_fn = nn.CrossEntropyLoss()

        # Metrics
        self.accuracy = Accuracy(task="binary")
        self.precision = Precision(task="binary")

        # Optimizer
        self.lr = learning_rate

    def forward(self, x):
        return self.model(x)

    def _sharable_step(self, batch, batch_idx):
        images, labels = batch
        logits = self.forward(images)
        loss = self.loss_fn(logits, labels)
        _, preds = torch.max(logits, 1)
        return loss, preds, labels

    def training_step(self, batch, batch_idx):
        loss, preds, labels = self._sharable_step(batch, batch_idx)
        self.log("train_loss", loss)
        self.log("train_acc", self.accuracy(preds, labels))
        self.log("train_precision", self.precision(preds, labels))
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, labels = self._sharable_step(batch, batch_idx)
        self.log("val_loss", loss)
        self.log("val_acc", self.accuracy(preds, labels), prog_bar=True)
        self.log("val_precision", self.precision(preds, labels), prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, preds, labels = self._sharable_step(batch, batch_idx)
        self.log("test_loss", loss)
        self.log("test_acc", self.accuracy(preds, labels))
        self.log("test_precision", self.precision(preds, labels))
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        images, _ = batch
        logits = self.forward(images)
        _, preds = torch.max(logits, 1)  # Get the predicted classes
        return preds
