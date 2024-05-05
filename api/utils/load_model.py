import torch
import torch.nn as nn
from torchvision import models


def load_model(num_classes: int = 2, model_weights: str = None):
    # load model architecture
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    # load state
    state_dict = torch.load(model_weights)
    adjusted_state_dict = {
        key.replace("model.", ""): value for key, value in state_dict.items()
    }
    model.load_state_dict(adjusted_state_dict)
    return model
