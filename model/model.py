import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics import Accuracy


class ClassificationModel(pl.LightningModule):
    def __init__(self, input_shape=(3, 256, 256), learning_rate: float = 0.01):
        super().__init__()
        self.lr = learning_rate
        self.loss_fn = nn.BCELoss()
        self.accuracy = Accuracy(task="binary")

        self.conv1 = nn.Conv2d(
            in_channels=input_shape[0],
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Example calculation for the output size after the conv and pool layers
        # After each pool, the size is halved if kernel_size=2 and stride=2
        # Adjust the calculation based on your network's architecture
        self.num_flat_features = (
            32 * (input_shape[1] // 4) * (input_shape[2] // 4)
        )  # Adjust this calculation

        self.fc1 = nn.Linear(self.num_flat_features, 64)
        self.fc2 = nn.Linear(64, 1)

        self.validation_targets = []
        self.validation_scores = []
        self.test_targets = []
        self.test_scores = []

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # Flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

    def _score_accuracy(self, true, predicted, log_name):
        accuracy = self.accuracy(true[0], predicted[0])
        self.log(log_name, accuracy)
        true.clear()
        predicted.clear()

    def _sharable_step(self, batch, batch_idx):
        images, labels = batch
        labels = labels.float()
        probabilities = self.forward(images).squeeze()
        predicted_labels = (probabilities > 0.5).float()
        loss = self.loss_fn(probabilities, labels)
        return loss, predicted_labels, labels

    def training_step(self, batch, batch_idx):
        loss, scores, y = self._sharable_step(batch, batch_idx)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, scores, y = self._sharable_step(batch, batch_idx)
        self.validation_targets.append(y)
        self.validation_scores.append(scores)
        self.log("val_loss", loss)
        return loss

    def on_validation_epoch_end(self):
        self._score_accuracy(
            self.validation_targets, self.validation_scores, "val_accuracy"
        )

    def test_step(self, batch, batch_idx):
        loss, scores, y = self._sharable_step(batch, batch_idx)
        self.test_targets.append(y)
        self.test_scores.append(scores)
        self.log("test_loss", loss)
        return loss

    def on_test_epoch_end(self):
        self._score_accuracy(self.test_targets, self.test_scores, "test_accuracy")

    def predict_step(self, batch, batch_idx):
        x, y = batch
        outputs = self.forward(x)
        outputs = (outputs > 0.5).int()
        return outputs

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)
