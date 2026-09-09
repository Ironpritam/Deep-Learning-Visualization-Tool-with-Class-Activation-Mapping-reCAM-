"""
Explainable AI (XAI) Engine: Vanilla CAM, Grad-CAM, and Grad-CAM++ implementations in pure PyTorch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.utils.imagenet_labels import get_label

SUPPORTED_XAI_METHODS = ["vanilla_cam", "gradcam", "gradcam_pp"]

def get_top_k_predictions(model: nn.Module, tensor_img: torch.Tensor, top_k: int = 5):
    """
    Computes model predictions and returns Top-K classes with probabilities and labels.
    """
    model.eval()
    device = next(model.parameters()).device
    tensor_img = tensor_img.to(device)
    with torch.no_grad():
        output = model(tensor_img)
        probs = F.softmax(output, dim=1).squeeze(0)

    top_probs, top_indices = torch.topk(probs, top_k)
    results = []
    for idx, prob in zip(top_indices.tolist(), top_probs.tolist()):
        results.append({
            "class_index": idx,
            "class_name": get_label(idx),
            "confidence": float(prob),
            "percentage": f"{prob * 100:.2f}%"
        })
    return results

class XAIEngine:
    """
    Unified Explainable AI Engine for Vanilla CAM, Grad-CAM, and Grad-CAM++.
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module, fc_weight: torch.Tensor = None):
        self.model = model
        self.target_layer = target_layer
        self.fc_weight = fc_weight
        self.device = next(model.parameters()).device
        self.activations = None
        self.gradients = None

        # Register PyTorch forward and backward hooks on target layer
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_vanilla_cam(self, tensor_img: torch.Tensor, target_class: int) -> np.ndarray:
        """
        Vanilla Class Activation Mapping (CAM).
        Calculates S_c(x,y) = ReLU( sum_k w_k^c * A_k(x,y) )
        """
        if self.fc_weight is None:
            raise ValueError("FC weights required for Vanilla CAM.")

        tensor_img = tensor_img.to(self.device)
        self.model.zero_grad()
        output = self.model(tensor_img)
        
        # Feature maps shape: (1, C, H, W)
        feature_maps = self.activations.squeeze(0)          # (C, H, W)
        weights = self.fc_weight[target_class].to(self.device) # (C,)

        # Linear combination of weights and feature maps
        cam = torch.matmul(weights, feature_maps.reshape(feature_maps.shape[0], -1))
        cam = cam.reshape(feature_maps.shape[1], feature_maps.shape[2])
        cam = F.relu(cam)

        # Normalize to [0, 1]
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        # Interpolate to 224x224
        cam_tensor = cam.unsqueeze(0).unsqueeze(0)
        cam_resized = F.interpolate(cam_tensor, size=(224, 224), mode='bilinear', align_corners=False)
        return cam_resized.squeeze().cpu().numpy()

    def generate_grad_cam(self, tensor_img: torch.Tensor, target_class: int) -> np.ndarray:
        """
        Grad-CAM: Gradient-weighted Class Activation Mapping.
        """
        tensor_img = tensor_img.to(self.device)
        self.model.zero_grad()
        output = self.model(tensor_img)
        score = output[0, target_class]
        score.backward()

        gradients = self.gradients[0]     # (C, H, W)
        activations = self.activations[0] # (C, H, W)

        # Global average pooling over gradients to get channel weights
        weights = torch.mean(gradients, dim=(1, 2))  # (C,)

        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=self.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = F.relu(cam)
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        cam_tensor = cam.unsqueeze(0).unsqueeze(0)
        cam_resized = F.interpolate(cam_tensor, size=(224, 224), mode='bilinear', align_corners=False)
        return cam_resized.squeeze().cpu().numpy()

    def generate_grad_cam_pp(self, tensor_img: torch.Tensor, target_class: int) -> np.ndarray:
        """
        Grad-CAM++: Generalized Gradient-weighted Class Activation Mapping.
        """
        tensor_img = tensor_img.to(self.device)
        self.model.zero_grad()
        output = self.model(tensor_img)
        score = output[0, target_class]
        score.backward()

        gradients = self.gradients[0]     # (C, H, W)
        activations = self.activations[0] # (C, H, W)

        g2 = gradients ** 2
        g3 = gradients ** 3
        sum_activations = torch.sum(activations, dim=(1, 2), keepdim=True)
        alpha_denom = 2 * g2 + sum_activations * g3
        alpha_denom = torch.where(alpha_denom != 0, alpha_denom, torch.ones_like(alpha_denom))
        alphas = g2 / alpha_denom

        weights = torch.sum(alphas * F.relu(gradients), dim=(1, 2))
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=self.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = F.relu(cam)
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        cam_tensor = cam.unsqueeze(0).unsqueeze(0)
        cam_resized = F.interpolate(cam_tensor, size=(224, 224), mode='bilinear', align_corners=False)
        return cam_resized.squeeze().cpu().numpy()

def generate_explanation(model: nn.Module, target_layer: nn.Module, fc_weight: torch.Tensor,
                         tensor_img: torch.Tensor, target_class: int, method: str = "gradcam") -> np.ndarray:
    """
    Unified function to generate CAM heatmap array for a target image and class.
    """
    engine = XAIEngine(model, target_layer, fc_weight)
    method = method.lower().strip()

    if method == "vanilla_cam":
        return engine.generate_vanilla_cam(tensor_img, target_class)
    elif method == "gradcam_pp":
        return engine.generate_grad_cam_pp(tensor_img, target_class)
    else:  # Default to Grad-CAM
        return engine.generate_grad_cam(tensor_img, target_class)

