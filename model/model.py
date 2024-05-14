import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics import Accuracy, Precision
from torchvision.models import (densenet121, inception_v3, mobilenet_v2, mobilenet_v3_large,
                                resnet18, resnet50, vgg16, squeezenet1_0)
from efficientnet_pytorch import EfficientNet


class ClassificationModel(pl.LightningModule):
    def __init__(self, learning_rate: float = 0.001):
        super().__init__()
        self.model = squeezenet1_0(pretrained=True)

        # squeezenet1_0
        self.model.classifier[1] = nn.Conv2d(512, 2, kernel_size=(1, 1))

        # vgg
        # num_features = self.model.classifier[6].in_features
        # self.model.classifier[6] = nn.Linear(num_features, 2) 

        # densenet
        # num_ftrs = self.model.classifier.in_features
        # self.model.classifier = nn.Linear(num_ftrs, 2)

        # resnet
        # num_ftrs = self.model.fc.in_features
        # self.model.fc = nn.Linear(num_ftrs, 2)

        # mobilenet v2
        # num_ftrs = self.model.classifier[1].in_features
        # self.model.classifier[1] = nn.Linear(num_ftrs, 2)
        
        # mobilenet v3
        # num_ftrs = self.model.classifier[3].in_features
        # self.model.classifier[3] = nn.Linear(num_ftrs, 2)

        # efficienet
        # self.model = EfficientNet.from_pretrained("efficientnet-b0", num_classes=2)

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
        _, preds = torch.max(logits, 1)
        return preds
