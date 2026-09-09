"""
Unified Command Line Interface and Application Launcher for Re-CAM
Developed at IIT Ropar
"""

import argparse
import sys
import os

def run_cli_inference(args):
    print("=" * 60)
    print(" Re-CAM Explainable AI (XAI) Suite — CLI Batch Engine")
    print(" Developed at IIT Ropar")
    print("=" * 60)

    from src.models.backbones import load_vision_backbone
    from src.xai.recam_engine import generate_explanation, get_top_k_predictions
    from src.utils.image_processing import (
        load_and_preprocess_image, apply_colormap, overlay_heatmap,
        extract_roi_bounding_boxes, draw_bounding_boxes, generate_side_by_side_canvas
    )
    from src.utils.imagenet_labels import get_label
    import cv2

    print(f"[*] Loading vision backbone model: {args.model}...")
    model, target_layer, fc_weight, device = load_vision_backbone(args.model)
    print(f"[*] Compute device selected: {device.type.upper()}")

    print(f"[*] Processing input image: {args.image}...")
    rgb_img_224, tensor_img = load_and_preprocess_image(args.image)

    print("[*] Computing Top-5 model predictions...")
    top_5 = get_top_k_predictions(model, tensor_img, top_k=5)
    print("\n--- Model Predictions ---")
    for i, pred in enumerate(top_5):
        print(f" #{i+1}: [{pred['class_index']}] {pred['class_name']} — {pred['percentage']}")

    target_class = args.target if args.target is not None else top_5[0]['class_index']
    target_label = get_label(target_class)
    print(f"\n[*] Generating {args.method.upper()} heatmap for Target Class [{target_class}] '{target_label}'...")

    cam_norm = generate_explanation(model, target_layer, fc_weight, tensor_img, target_class, method=args.method)

    heatmap_rgb = apply_colormap(cam_norm, colormap_name=args.colormap)
    blended = overlay_heatmap(rgb_img_224, heatmap_rgb, alpha=args.alpha)
    roi_boxes = extract_roi_bounding_boxes(cam_norm, threshold_ratio=args.roi_thresh)
    overlay_roi_rgb = draw_bounding_boxes(blended, roi_boxes)

    side_by_side = generate_side_by_side_canvas(rgb_img_224, heatmap_rgb, overlay_roi_rgb)

    output_path = args.output if args.output else "recam_output.jpg"
    cv2.imwrite(output_path, cv2.cvtColor(side_by_side, cv2.COLOR_RGB2BGR))
    print(f"\n[+] Success! Side-by-Side visualization saved to: {os.path.abspath(output_path)}")
    print(f"[+] Localized ROI Bounding Boxes detected: {len(roi_boxes)}")

def main():
    parser = argparse.ArgumentParser(description="Re-CAM Explainable AI (XAI) Suite — IIT Ropar")
    parser.add_argument("--gui", action="store_true", help="Launch Desktop GUI Interface")
    parser.add_argument("--api", action="store_true", help="Launch FastAPI REST Microservice Server")
    
    # CLI options
    parser.add_argument("--image", type=str, help="Path to input image for CLI inference")
    parser.add_argument("--model", type=str, default="resnet50", choices=["resnet50", "resnet18", "vgg16", "mobilenet_v2"], help="Vision backbone model")
    parser.add_argument("--method", type=str, default="gradcam", choices=["vanilla_cam", "gradcam", "gradcam_pp"], help="XAI algorithm method")
    parser.add_argument("--target", type=int, default=None, help="Target ImageNet Class Index (Default: Top-1 predicted class)")
    parser.add_argument("--colormap", type=str, default="JET", help="OpenCV Colormap name (JET, VIRIDIS, PLASMA, INFERNO, HOT, RAINBOW)")
    parser.add_argument("--alpha", type=float, default=0.5, help="Heatmap alpha blending transparency (0.0 to 1.0)")
    parser.add_argument("--roi-thresh", type=float, default=0.7, help="Activation intensity threshold for ROI bounding box extraction")
    parser.add_argument("--output", type=str, default="recam_output.jpg", help="Path to save generated side-by-side visualization canvas")

    args = parser.parse_args()

    if args.gui:
        import app_gui
        root = app_gui.tk.Tk()
        app = app_gui.ReCAMApp(root)
        root.mainloop()
    elif args.api:
        import uvicorn
        print("Starting Re-CAM FastAPI Microservice on http://localhost:8000 ...")
        uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)
    elif args.image:
        run_cli_inference(args)
    else:
        # Default behavior if no flags: Launch GUI
        print("No mode specified. Launching Desktop GUI (Use --help for CLI / API options)...")
        import app_gui
        root = app_gui.tk.Tk()
        app = app_gui.ReCAMApp(root)
        root.mainloop()

if __name__ == "__main__":
    main()
