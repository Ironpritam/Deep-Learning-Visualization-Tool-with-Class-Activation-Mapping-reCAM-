"""
FastAPI RESTful Microservice Endpoint for Re-CAM Explainable AI Suite
Developed at IIT Ropar
"""

import base64
import io
import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import Optional, List

from src.models.backbones import load_vision_backbone, SUPPORTED_MODELS
from src.xai.recam_engine import generate_explanation, get_top_k_predictions, SUPPORTED_XAI_METHODS
from src.utils.image_processing import (
    load_and_preprocess_image, apply_colormap, overlay_heatmap,
    extract_roi_bounding_boxes, draw_bounding_boxes, generate_side_by_side_canvas
)
from src.utils.imagenet_labels import get_label

app = FastAPI(
    title="Re-CAM Explainable AI (XAI) Microservice",
    description="Production RESTful API for CNN Model Diagnostics, Class Activation Maps (CAM/Grad-CAM), and Automated ROI Extraction.",
    version="2.0.0"
)

# Model cache dictionary
MODEL_CACHE = {}

def get_cached_model(model_name: str):
    model_name = model_name.lower().strip()
    if model_name not in MODEL_CACHE:
        MODEL_CACHE[model_name] = load_vision_backbone(model_name)
    return MODEL_CACHE[model_name]

class PredictionItem(BaseModel):
    class_index: int
    class_name: str
    confidence: float
    percentage: str

class BoundingBox(BaseModel):
    x: int
    y: int
    w: int
    h: int

class ExplainResponse(BaseModel):
    status: str
    model: str
    xai_method: str
    target_class_index: int
    target_class_name: str
    top_5_predictions: List[PredictionItem]
    roi_bounding_boxes: List[BoundingBox]
    canvas_base64: str

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "Re-CAM XAI Microservice",
        "institution": "IIT Ropar",
        "supported_models": SUPPORTED_MODELS,
        "supported_xai_methods": SUPPORTED_XAI_METHODS
    }

@app.get("/api/v1/models")
def get_models():
    return {
        "models": SUPPORTED_MODELS,
        "xai_methods": SUPPORTED_XAI_METHODS
    }

@app.post("/api/v1/explain", response_model=ExplainResponse)
async def explain_image(
    file: UploadFile = File(...),
    model_name: str = Form("resnet50"),
    xai_method: str = Form("gradcam"),
    target_class: Optional[int] = Form(None),
    colormap: str = Form("JET"),
    alpha: float = Form(0.5),
    roi_threshold: float = Form(0.7)
):
    try:
        image_bytes = await file.read()
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        rgb_img_224, tensor_img = load_and_preprocess_image(pil_img)

        # Load backbone
        model, target_layer, fc_weight, device = get_cached_model(model_name)

        # Get Top 5 predictions
        top_5 = get_top_k_predictions(model, tensor_img, top_k=5)

        # If target class not specified, use Top-1 prediction
        if target_class is None:
            target_class = top_5[0]['class_index']

        target_name = get_label(target_class)

        # Generate CAM Heatmap
        cam_norm = generate_explanation(model, target_layer, fc_weight, tensor_img, target_class, method=xai_method)

        # Render visualizations
        heatmap_rgb = apply_colormap(cam_norm, colormap_name=colormap)
        blended = overlay_heatmap(rgb_img_224, heatmap_rgb, alpha=alpha)
        
        raw_boxes = extract_roi_bounding_boxes(cam_norm, threshold_ratio=roi_threshold)
        overlay_roi_rgb = draw_bounding_boxes(blended, raw_boxes)

        side_by_side = generate_side_by_side_canvas(rgb_img_224, heatmap_rgb, overlay_roi_rgb)

        # Encode canvas image to Base64
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(side_by_side, cv2.COLOR_RGB2BGR))
        canvas_b64 = base64.b64encode(buffer).decode('utf-8')

        formatted_boxes = [BoundingBox(x=b[0], y=b[1], w=b[2], h=b[3]) for b in raw_boxes]
        formatted_top5 = [PredictionItem(**p) for p in top_5]

        return ExplainResponse(
            status="success",
            model=model_name,
            xai_method=xai_method,
            target_class_index=target_class,
            target_class_name=target_name,
            top_5_predictions=formatted_top5,
            roi_bounding_boxes=formatted_boxes,
            canvas_base64=f"data:image/jpeg;base64,{canvas_b64}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
