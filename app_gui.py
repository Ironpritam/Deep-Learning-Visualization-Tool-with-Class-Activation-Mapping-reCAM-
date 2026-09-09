"""
Modern Desktop GUI for Re-CAM Explainable AI (XAI) Suite
Developed at IIT Ropar
"""

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk

from src.models.backbones import load_vision_backbone, SUPPORTED_MODELS
from src.xai.recam_engine import generate_explanation, get_top_k_predictions, SUPPORTED_XAI_METHODS
from src.utils.image_processing import (
    load_and_preprocess_image, apply_colormap, overlay_heatmap,
    extract_roi_bounding_boxes, draw_bounding_boxes, generate_side_by_side_canvas, COLORMAPS
)
from src.utils.imagenet_labels import get_label, search_classes
from src.reporting.report_generator import generate_pdf_report

class ReCAMApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Re-CAM: Deep Learning Visualization & Explainable AI Suite (IIT Ropar)")
        self.root.geometry("1180x760")
        self.root.configure(bg="#1A202C")

        # System state variables
        self.current_image_path = None
        self.rgb_img_224 = None
        self.tensor_img = None
        self.model = None
        self.target_layer = None
        self.fc_weight = None
        self.current_model_name = None

        self.cam_norm = None
        self.current_canvas_rgb = None
        self.top_5_preds = []

        self.selected_model_var = tk.StringVar(value="resnet50")
        self.selected_xai_var = tk.StringVar(value="gradcam")
        self.selected_cmap_var = tk.StringVar(value="JET")
        self.target_class_var = tk.StringVar(value="281")
        self.alpha_var = tk.DoubleVar(value=0.5)
        self.roi_thresh_var = tk.DoubleVar(value=0.7)

        self._build_ui()
        self._load_selected_model()

    def _build_ui(self):
        # Apply dark theme styling
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TFrame", background="#1A202C")
        style.configure("TLabelFrame", background="#2D3748", foreground="#FFFFFF", font=("Segoe UI", 10, "bold"))
        style.configure("TLabel", background="#2D3748", foreground="#E2E8F0", font=("Segoe UI", 9))
        style.configure("TButton", font=("Segoe UI", 9, "bold"), background="#3182CE", foreground="#FFFFFF")
        style.map("TButton", background=[('active', '#2B6CB0')])

        # Top Title Banner
        banner_frame = tk.Frame(self.root, bg="#2B6CB0", height=45)
        banner_frame.pack(side="top", fill="x")
        title_label = tk.Label(banner_frame, text="Re-CAM | Explainable AI & Model Diagnostics (IIT Ropar)",
                               font=("Segoe UI", 13, "bold"), bg="#2B6CB0", fg="#FFFFFF")
        title_label.pack(side="left", padx=15, pady=8)

        # Main Layout Container: Control Panel (Left) + Visualization Canvas (Right)
        main_container = tk.Frame(self.root, bg="#1A202C")
        main_container.pack(expand=True, fill="both", padx=10, pady=10)

        # Left Control Panel
        control_panel = tk.Frame(main_container, bg="#2D3748", width=340)
        control_panel.pack(side="left", fill="y", padx=(0, 10))

        # --- Section 1: Model & XAI Selector ---
        sec1 = ttk.LabelFrame(control_panel, text=" 1. Architecture & XAI Engine ")
        sec1.pack(fill="x", padx=10, pady=8)

        tk.Label(sec1, text="Vision Backbone:", bg="#2D3748", fg="#E2E8F0").grid(row=0, column=0, sticky="w", padx=5, pady=4)
        model_cb = ttk.Combobox(sec1, textvariable=self.selected_model_var, values=SUPPORTED_MODELS, state="readonly", width=14)
        model_cb.grid(row=0, column=1, padx=5, pady=4)
        model_cb.bind("<<ComboboxSelected>>", lambda e: self._load_selected_model())

        tk.Label(sec1, text="XAI Algorithm:", bg="#2D3748", fg="#E2E8F0").grid(row=1, column=0, sticky="w", padx=5, pady=4)
        xai_cb = ttk.Combobox(sec1, textvariable=self.selected_xai_var, values=SUPPORTED_XAI_METHODS, state="readonly", width=14)
        xai_cb.grid(row=1, column=1, padx=5, pady=4)

        # --- Section 2: Input Image & Class Selection ---
        sec2 = ttk.LabelFrame(control_panel, text=" 2. Input Image & Class ")
        sec2.pack(fill="x", padx=10, pady=8)

        load_btn = ttk.Button(sec2, text="📁 Load Input Image", command=self._open_image)
        load_btn.pack(fill="x", padx=5, pady=6)

        class_frame = tk.Frame(sec2, bg="#2D3748")
        class_frame.pack(fill="x", padx=5, pady=4)
        tk.Label(class_frame, text="Target Class ID:", bg="#2D3748", fg="#E2E8F0").pack(side="left")
        class_entry = ttk.Entry(class_frame, textvariable=self.target_class_var, width=8)
        class_entry.pack(side="left", padx=5)

        # Top Predictions Listbox
        tk.Label(sec2, text="Top-5 Predicted Classes:", bg="#2D3748", fg="#CBD5E0", font=("Segoe UI", 8, "italic")).pack(anchor="w", padx=5, pady=(5,0))
        self.preds_listbox = tk.Listbox(sec2, height=5, bg="#1A202C", fg="#63B3ED", font=("Segoe UI", 8), selectbackground="#3182CE")
        self.preds_listbox.pack(fill="x", padx=5, pady=4)
        self.preds_listbox.bind("<<ListboxSelect>>", self._on_select_prediction)

        # --- Section 3: Visual Styling Controls ---
        sec3 = ttk.LabelFrame(control_panel, text=" 3. Visualization Tuning ")
        sec3.pack(fill="x", padx=10, pady=8)

        tk.Label(sec3, text="Colormap:", bg="#2D3748", fg="#E2E8F0").grid(row=0, column=0, sticky="w", padx=5, pady=4)
        cmap_cb = ttk.Combobox(sec3, textvariable=self.selected_cmap_var, values=list(COLORMAPS.keys()), state="readonly", width=12)
        cmap_cb.grid(row=0, column=1, padx=5, pady=4)

        tk.Label(sec3, text="Alpha Transparency:", bg="#2D3748", fg="#E2E8F0").grid(row=1, column=0, sticky="w", padx=5, pady=4)
        alpha_scale = ttk.Scale(sec3, from_=0.1, to=1.0, variable=self.alpha_var, orient="horizontal", command=lambda v: self._update_rendering())
        alpha_scale.grid(row=1, column=1, padx=5, pady=4)

        tk.Label(sec3, text="ROI Threshold:", bg="#2D3748", fg="#E2E8F0").grid(row=2, column=0, sticky="w", padx=5, pady=4)
        roi_scale = ttk.Scale(sec3, from_=0.4, to=0.9, variable=self.roi_thresh_var, orient="horizontal", command=lambda v: self._update_rendering())
        roi_scale.grid(row=2, column=1, padx=5, pady=4)

        # --- Section 4: Actions & Export ---
        sec4 = ttk.LabelFrame(control_panel, text=" 4. Action & Export ")
        sec4.pack(fill="x", padx=10, pady=8)

        gen_btn = ttk.Button(sec4, text="⚡ Generate Heatmap", command=self._generate_explanation)
        gen_btn.pack(fill="x", padx=5, pady=4)

        save_img_btn = ttk.Button(sec4, text="💾 Save Canvas Image", command=self._save_canvas)
        save_img_btn.pack(fill="x", padx=5, pady=4)

        pdf_btn = ttk.Button(sec4, text="📄 Export PDF Report", command=self._export_pdf)
        pdf_btn.pack(fill="x", padx=5, pady=4)

        # Right Display Area (Side-by-Side Visualization)
        display_frame = tk.Frame(main_container, bg="#2D3748")
        display_frame.pack(side="right", expand=True, fill="both")

        self.canvas_label = tk.Label(display_frame, bg="#1A202C", text="Please load an image and click 'Generate Heatmap'",
                                     fg="#A0AEC0", font=("Segoe UI", 11))
        self.canvas_label.pack(expand=True, fill="both", padx=10, pady=10)

    def _load_selected_model(self):
        model_name = self.selected_model_var.get()
        try:
            self.model, self.target_layer, self.fc_weight, _ = load_vision_backbone(model_name)
            self.current_model_name = model_name
            if self.tensor_img is not None:
                self._update_top_predictions()
        except Exception as e:
            messagebox.showerror("Model Load Error", f"Failed to load backbone model {model_name}:\n{e}")

    def _open_image(self):
        filepath = filedialog.askopenfilename(title="Select Input Image",
                                              filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")])
        if filepath:
            try:
                self.current_image_path = filepath
                self.rgb_img_224, self.tensor_img = load_and_preprocess_image(filepath)
                self._update_top_predictions()
                messagebox.showinfo("Success", f"Loaded image successfully: {os.path.basename(filepath)}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image:\n{e}")

    def _update_top_predictions(self):
        if self.model is None or self.tensor_img is None:
            return
        self.top_5_preds = get_top_k_predictions(self.model, self.tensor_img, top_k=5)
        self.preds_listbox.delete(0, tk.END)
        for pred in self.top_5_preds:
            self.preds_listbox.insert(tk.END, f"[{pred['class_index']}] {pred['class_name']} ({pred['percentage']})")
        
        # Set default target class to top-1 prediction
        if self.top_5_preds:
            top_1_idx = self.top_5_preds[0]['class_index']
            self.target_class_var.set(str(top_1_idx))

    def _on_select_prediction(self, event):
        selection = self.preds_listbox.curselection()
        if selection:
            index = selection[0]
            selected_pred = self.top_5_preds[index]
            self.target_class_var.set(str(selected_pred['class_index']))
            self._generate_explanation()

    def _generate_explanation(self):
        if self.tensor_img is None:
            messagebox.showwarning("Warning", "Please load an image first!")
            return
        try:
            target_class = int(self.target_class_var.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid integer for Target Class ID!")
            return

        xai_method = self.selected_xai_var.get()

        try:
            self.cam_norm = generate_explanation(
                self.model, self.target_layer, self.fc_weight,
                self.tensor_img, target_class, method=xai_method
            )
            self._update_rendering()
        except Exception as e:
            messagebox.showerror("Execution Error", f"Heatmap generation failed:\n{e}")

    def _update_rendering(self):
        if self.cam_norm is None or self.rgb_img_224 is None:
            return

        cmap_name = self.selected_cmap_var.get()
        alpha = float(self.alpha_var.get())
        roi_thresh = float(self.roi_thresh_var.get())

        heatmap_rgb = apply_colormap(self.cam_norm, cmap_name)
        blended = overlay_heatmap(self.rgb_img_224, heatmap_rgb, alpha=alpha)
        
        # Extract and draw ROI boxes
        roi_boxes = extract_roi_bounding_boxes(self.cam_norm, threshold_ratio=roi_thresh)
        overlay_roi_rgb = draw_bounding_boxes(blended, roi_boxes)

        # Generate 3-panel side-by-side canvas
        self.current_canvas_rgb = generate_side_by_side_canvas(self.rgb_img_224, heatmap_rgb, overlay_roi_rgb)

        # Resize for GUI display
        pil_canvas = Image.fromarray(self.current_canvas_rgb)
        pil_canvas_resized = pil_canvas.resize((760, 260), Image.Resampling.LANCZOS)
        
        self.tk_canvas_img = ImageTk.PhotoImage(pil_canvas_resized)
        self.canvas_label.config(image=self.tk_canvas_img, text="")

    def _save_canvas(self):
        if self.current_canvas_rgb is None:
            messagebox.showwarning("Warning", "No canvas output available to save!")
            return
        filepath = filedialog.asksaveasfilename(defaultextension=".jpg",
                                                filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")])
        if filepath:
            cv2.imwrite(filepath, cv2.cvtColor(self.current_canvas_rgb, cv2.COLOR_RGB2BGR))
            messagebox.showinfo("Saved", f"Canvas saved to {filepath}")

    def _export_pdf(self):
        if self.current_canvas_rgb is None:
            messagebox.showwarning("Warning", "No heatmap generated to export!")
            return
        filepath = filedialog.asksaveasfilename(defaultextension=".pdf",
                                                filetypes=[("PDF Document", "*.pdf")])
        if filepath:
            target_idx = int(self.target_class_var.get())
            target_info = {
                "class_index": target_idx,
                "class_name": get_label(target_idx),
                "percentage": next((p['percentage'] for p in self.top_5_preds if p['class_index'] == target_idx), "N/A")
            }
            roi_boxes = extract_roi_bounding_boxes(self.cam_norm, threshold_ratio=float(self.roi_thresh_var.get()))
            generate_pdf_report(
                filepath, self.current_canvas_rgb, self.current_model_name,
                self.selected_xai_var.get(), target_info, self.top_5_preds, roi_boxes
            )
            messagebox.showinfo("Export Success", f"PDF Report saved to {filepath}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ReCAMApp(root)
    root.mainloop()
