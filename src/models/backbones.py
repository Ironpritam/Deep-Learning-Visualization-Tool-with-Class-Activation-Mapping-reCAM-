"""
Multi-Model Vision Backbone Loader for PyTorch (ResNet-50, ResNet-18, VGG-16, MobileNet-V2)
"""

import torch
import torch.nn as nn
import torchvision.models as models

SUPPORTED_MODELS = ["resnet50", "resnet18", "vgg16", "mobilenet_v2"]

def load_vision_backbone(model_name: str = "resnet50", pretrained: bool = True):
    """
    Loads pre-trained PyTorch backbone and identifies target convolutional layer for CAM.
    Returns:
        model: PyTorch nn.Module in eval mode
        target_layer: nn.Module target conv layer for feature map hooks
        fc_weight: Tensor weight matrix of classifier layer (num_classes, in_features)
    """
    model_name = model_name.lower().strip()
    weights = "DEFAULT" if pretrained else None

    if model_name == "resnet50":
        model = models.resnet50(weights=weights)
        target_layer = model.layer4[-1]
        fc_weight = model.fc.weight.data
    elif model_name == "resnet18":
        model = models.resnet18(weights=weights)
        target_layer = model.layer4[-1]
        fc_weight = model.fc.weight.data
    elif model_name == "vgg16":
        model = models.vgg16(weights=weights)
        target_layer = model.features[28]  # Final conv layer of VGG16
        # VGG fc weights derived from classifier linear layer
        fc_weight = model.classifier[6].weight.data
    elif model_name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=weights)
        target_layer = model.features[-1]
        fc_weight = model.classifier[1].weight.data
    else:
        raise ValueError(f"Model '{model_name}' not supported. Choose from {SUPPORTED_MODELS}")

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model, target_layer, fc_weight, device

