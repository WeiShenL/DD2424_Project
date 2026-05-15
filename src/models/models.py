import torch.nn as nn
from torchvision import models

def build_model(device, num_classes=2) -> models.ResNet:
    model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False

    # Replace the final layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)


def build_resnet18_model(device, num_classes=2) -> models.ResNet:
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False

    # Replace the final layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)

def build_model_finetune(device, num_classes=37, l=1) -> models.ResNet:
    model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    
    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the last l layers based on l
    layers_to_unfreeze = ['layer4', 'layer3', 'layer2', 'layer1'][:l]
    for layer_name in layers_to_unfreeze:
        for param in getattr(model, layer_name).parameters():
            param.requires_grad = True

    # Always replace and unfreeze fc
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)

def unfreeze_layer(model, layer_name):
    """Unfreeze a specific layer by name."""
    for param in getattr(model, layer_name).parameters():
        param.requires_grad = True
    print(f"  → Unfreezing {layer_name}")