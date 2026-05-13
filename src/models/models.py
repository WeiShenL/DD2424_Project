import torch.nn as nn
from torchvision import models

def build_model(device, num_classes=2) -> models.ResNet:
    model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False

    # Replace the final layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)