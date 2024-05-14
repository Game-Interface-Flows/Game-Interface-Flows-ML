import torch
import torch.nn as nn
from torchvision import models


def load_model(num_classes: int = 2, model_weights: str = None):
    # load model architecture
    model = models.mobilenet_v2(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    # load weights
    state_dict = torch.load(model_weights)
    weights = {key.replace("model.", ""): value for key, value in state_dict.items()}
    model.load_state_dict(weights)
    return model
