"""
Diagnostic Report Generator for Explainable AI (XAI) Audit Trails.
Generates PDF reports containing visual heatmaps, top-5 predictions, and ROI coordinates.
"""

import os
import cv2
import numpy as np
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

def generate_pdf_report(
    output_pdf_path: str,
    canvas_rgb: np.ndarray,
    model_name: str,
    xai_method: str,
    target_class_info: dict,
    top_5_preds: list,
    bounding_boxes: list
):
    """
    Generates a structured PDF report for XAI audit trails.
    """
    doc = SimpleDocTemplate(output_pdf_path, pagesize=letter,
                            rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    story = []
    styles = getSampleStyleSheet()

    # Custom Header Style
    header_style = ParagraphStyle(
        'HeaderStyle',
        parent=styles['Heading1'],
        fontSize=20,
        textColor=colors.HexColor("#1A365D"),
        spaceAfter=6
    )
    subheader_style = ParagraphStyle(
        'SubHeaderStyle',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.HexColor("#4A5568"),
        spaceAfter=15
    )

    story.append(Paragraph("Re-CAM: Model Explainability Audit Report", header_style))
    story.append(Paragraph("Department of Artificial Intelligence | IIT Ropar", subheader_style))
    story.append(Spacer(1, 10))

    # Save Canvas Image temporarily for ReportLab
    temp_img_path = output_pdf_path + "_temp_canvas.png"
    cv2.imwrite(temp_img_path, cv2.cvtColor(canvas_rgb, cv2.COLOR_RGB2BGR))
    story.append(RLImage(temp_img_path, width=540, height=180))
    story.append(Spacer(1, 15))

    # Section: Execution Metadata
    story.append(Paragraph("<b>System & Execution Metadata</b>", styles['Heading2']))
    meta_data = [
        ["Backbone Model:", model_name.upper(), "XAI Method:", xai_method.upper()],
        ["Target Class Index:", str(target_class_info.get("class_index")), "Target Class Name:", target_class_info.get("class_name")],
        ["Target Confidence:", target_class_info.get("percentage"), "Detected ROI Boxes:", str(len(bounding_boxes))]
    ]
    meta_table = Table(meta_data, colWidths=[120, 150, 120, 150])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F7FAFC")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('PADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 15))

    # Section: Top 5 Class Predictions
    story.append(Paragraph("<b>Top-5 Model Predictions</b>", styles['Heading2']))
    pred_headers = [["Rank", "Class Index", "Class Label", "Confidence Score"]]
    for i, p in enumerate(top_5_preds):
        pred_headers.append([f"#{i+1}", str(p['class_index']), p['class_name'], p['percentage']])
    
    pred_table = Table(pred_headers, colWidths=[50, 80, 260, 150])
    pred_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#2B6CB0")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('PADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(pred_table)

    doc.build(story)

    # Clean up temp image file
    if os.path.exists(temp_img_path):
        os.remove(temp_img_path)

    return output_pdf_path
