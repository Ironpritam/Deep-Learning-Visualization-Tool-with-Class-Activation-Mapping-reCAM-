"""
Image Preprocessing, Normalization, Colormaps, Alpha Blending, and ROI Bounding Box Extraction.
"""

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

# Standard PyTorch ImageNet Transforms
standard_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

COLORMAPS = {
    "JET": cv2.COLORMAP_JET,
    "VIRIDIS": cv2.COLORMAP_VIRIDIS,
    "PLASMA": cv2.COLORMAP_PLASMA,
    "INFERNO": cv2.COLORMAP_INFERNO,
    "HOT": cv2.COLORMAP_HOT,
    "RAINBOW": cv2.COLORMAP_RAINBOW
}

def load_and_preprocess_image(image_input):
    """
    Loads an image from path, PIL Image, or numpy array.
    Returns:
        rgb_img_224: np.ndarray (224, 224, 3) uint8 RGB format
        tensor_img: torch.Tensor (1, 3, 224, 224) normalized PyTorch float tensor
    """
    if isinstance(image_input, str):
        pil_img = Image.open(image_input).convert('RGB')
    elif isinstance(image_input, Image.Image):
        pil_img = image_input.convert('RGB')
    elif isinstance(image_input, np.ndarray):
        if image_input.shape[2] == 3:
            pil_img = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
        else:
            pil_img = Image.fromarray(image_input).convert('RGB')
    else:
        raise ValueError("Unsupported image input format.")

    # Resize PIL image for visualization array
    pil_resized = pil_img.resize((224, 224))
    rgb_img_224 = np.array(pil_resized)

    # Transform tensor for PyTorch backbone input
    tensor_img = standard_transform(pil_resized).unsqueeze(0)

    return rgb_img_224, tensor_img

def apply_colormap(cam_norm: np.ndarray, colormap_name: str = "JET") -> np.ndarray:
    """
    Applies an OpenCV colormap to a normalized [0, 1] 2D heatmap array.
    Returns RGB uint8 image (224, 224, 3).
    """
    cam_uint8 = np.uint8(255 * np.clip(cam_norm, 0, 1))
    cmap_code = COLORMAPS.get(colormap_name.upper(), cv2.COLORMAP_JET)
    heatmap_bgr = cv2.applyColorMap(cam_uint8, cmap_code)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    return heatmap_rgb

def overlay_heatmap(rgb_img: np.ndarray, heatmap_rgb: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Blends original RGB image and RGB heatmap with alpha transparency.
    """
    blended = cv2.addWeighted(rgb_img, 1.0 - alpha, heatmap_rgb, alpha, 0)
    return blended

def extract_roi_bounding_boxes(cam_norm: np.ndarray, threshold_ratio: float = 0.7):
    """
    Extracts bounding boxes around high-activation regions (e.g. > 70% max intensity).
    Returns list of bounding boxes [(x, y, w, h), ...]
    """
    cam_uint8 = np.uint8(255 * np.clip(cam_norm, 0, 1))
    threshold_val = int(255 * threshold_ratio)
    _, thresh_map = cv2.threshold(cam_uint8, threshold_val, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(thresh_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 10 and h > 10:  # Filter out tiny noise contours
            boxes.append((x, y, w, h))
    return boxes

def draw_bounding_boxes(img_rgb: np.ndarray, boxes: list, color=(0, 255, 0), thickness=2) -> np.ndarray:
    """
    Draws bounding boxes onto an RGB image.
    """
    annotated = img_rgb.copy()
    for (x, y, w, h) in boxes:
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, thickness)
        cv2.putText(annotated, "ROI High Activation", (x, max(12, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return annotated

def generate_side_by_side_canvas(rgb_img: np.ndarray, heatmap_rgb: np.ndarray, overlay_roi_rgb: np.ndarray) -> np.ndarray:
    """
    Combines [Original Image] | [Heatmap] | [Overlay + ROI] horizontally.
    """
    h, w, c = rgb_img.shape
    banner_h = 30
    combined_w = w * 3
    combined_h = h + banner_h

    canvas = np.zeros((combined_h, combined_w, c), dtype=np.uint8)

    # Fill images
    canvas[banner_h:, :w] = rgb_img
    canvas[banner_h:, w:2*w] = heatmap_rgb
    canvas[banner_h:, 2*w:] = overlay_roi_rgb

    # Draw top labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, "Original Image", (10, 20), font, 0.5, (255, 255, 255), 1)
    cv2.putText(canvas, "Raw Heatmap", (w + 10, 20), font, 0.5, (255, 255, 255), 1)
    cv2.putText(canvas, "Overlay + ROI BBox", (2*w + 10, 20), font, 0.5, (255, 255, 255), 1)

    return canvas
