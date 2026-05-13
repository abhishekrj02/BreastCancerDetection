import io
import os
import json
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, HRFlowable
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT


PINK = colors.HexColor("#E91E8C")
BLUE = colors.HexColor("#1A2940")
GREEN = colors.HexColor("#27AE60")
RED = colors.HexColor("#E74C3C")
LIGHT_GRAY = colors.HexColor("#F5F7FA")


def generate_report(prediction, user, img_path=None, heatmap_path=None):
    """
    Generate a clinical-style PDF report for a prediction.
    Returns BytesIO buffer.
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        rightMargin=2 * cm, leftMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "Title", parent=styles["Heading1"],
        textColor=BLUE, fontSize=20, spaceAfter=4, alignment=TA_CENTER
    )
    subtitle_style = ParagraphStyle(
        "Subtitle", parent=styles["Normal"],
        textColor=PINK, fontSize=11, spaceAfter=2, alignment=TA_CENTER
    )
    section_style = ParagraphStyle(
        "Section", parent=styles["Heading2"],
        textColor=BLUE, fontSize=13, spaceBefore=12, spaceAfter=6
    )
    body_style = ParagraphStyle(
        "Body", parent=styles["Normal"],
        textColor=colors.HexColor("#2C3E50"), fontSize=9.5, leading=15
    )
    small_style = ParagraphStyle(
        "Small", parent=styles["Normal"],
        textColor=colors.HexColor("#7F8C8D"), fontSize=8.5, alignment=TA_CENTER
    )

    elements = []

    # ── Header ──
    elements.append(Paragraph("BreastGuard AI", title_style))
    elements.append(Paragraph("AI-Powered Mammogram Analysis Report", subtitle_style))
    elements.append(HRFlowable(width="100%", thickness=2, color=PINK, spaceAfter=12))

    # ── Patient & Report Info ──
    report_data = [
        ["Patient", user.username, "Report Date", datetime.now().strftime("%d %b %Y, %H:%M")],
        ["Email", user.email, "Scan Date", prediction.created_at.strftime("%d %b %Y, %H:%M")],
        ["Role", user.role.capitalize(), "Report ID", f"#BGD-{prediction.id:05d}"],
    ]
    info_table = Table(report_data, colWidths=[3 * cm, 6 * cm, 3 * cm, 6 * cm])
    info_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT_GRAY),
        ("TEXTCOLOR", (0, 0), (0, -1), BLUE),
        ("TEXTCOLOR", (2, 0), (2, -1), BLUE),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [LIGHT_GRAY, colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E5E5E5")),
        ("PADDING", (0, 0), (-1, -1), 6),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    elements.append(info_table)
    elements.append(Spacer(1, 16))

    # ── Images ──
    if img_path and os.path.exists(img_path) and heatmap_path and os.path.exists(heatmap_path):
        elements.append(Paragraph("Mammogram Analysis", section_style))
        img_table_data = [[
            RLImage(img_path, width=7.5 * cm, height=7.5 * cm),
            RLImage(heatmap_path, width=7.5 * cm, height=7.5 * cm),
        ], [
            Paragraph("Original Mammogram", small_style),
            Paragraph("Grad-CAM Heatmap (AI Focus Areas)", small_style),
        ]]
        img_table = Table(img_table_data, colWidths=[8.5 * cm, 8.5 * cm])
        img_table.setStyle(TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("PADDING", (0, 0), (-1, -1), 8),
            ("BACKGROUND", (0, 0), (-1, 0), LIGHT_GRAY),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E5E5E5")),
            ("ROUNDEDCORNERS", [4, 4, 4, 4]),
        ]))
        elements.append(img_table)
        elements.append(Spacer(1, 16))

    # ── Results ──
    elements.append(Paragraph("Detection Results", section_style))
    diag_color = GREEN if prediction.diagnosis == "Benign" else RED
    results_data = [
        ["Metric", "Value", "Interpretation"],
        ["Primary Diagnosis", prediction.diagnosis,
         "Non-cancerous tissue" if prediction.diagnosis == "Benign" else "Potential malignancy detected"],
        ["Benign Probability", f"{prediction.benign:.1f}%",
         "Higher is better — non-cancerous likelihood"],
        ["Malignant Probability", f"{prediction.malignant:.1f}%",
         "Lower is better — cancerous cell likelihood"],
    ]
    results_table = Table(results_data, colWidths=[5 * cm, 4 * cm, 9 * cm])
    results_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BLUE),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_GRAY]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E5E5E5")),
        ("PADDING", (0, 0), (-1, -1), 7),
        ("ALIGN", (1, 1), (1, -1), "CENTER"),
        ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
        ("TEXTCOLOR", (1, 1), (1, 1), diag_color),
        ("FONTNAME", (1, 1), (1, 1), "Helvetica-Bold"),
    ]))
    elements.append(results_table)
    elements.append(Spacer(1, 12))

    # ── Detailed Breakdown ──
    if prediction.detailed:
        elements.append(Paragraph("Density Class Breakdown", section_style))
        try:
            detailed = json.loads(prediction.detailed)
            detail_rows = [["Class", "Probability", "Type"]]
            for name, val in detailed.items():
                ctype = "Benign" if "benign" in name.lower() else "Malignant"
                detail_rows.append([name, f"{val:.2f}%", ctype])
            detail_table = Table(detail_rows, colWidths=[9 * cm, 4 * cm, 5 * cm])
            detail_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), BLUE),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_GRAY]),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E5E5E5")),
                ("PADDING", (0, 0), (-1, -1), 6),
                ("ALIGN", (1, 1), (2, -1), "CENTER"),
            ]))
            elements.append(detail_table)
            elements.append(Spacer(1, 12))
        except Exception:
            pass

    # ── AI Summary ──
    if prediction.ai_summary:
        elements.append(Paragraph("AI Health Summary", section_style))
        # Strip markdown and render as plain paragraphs
        summary_text = prediction.ai_summary
        for line in summary_text.split("\n"):
            line = line.strip()
            if not line:
                elements.append(Spacer(1, 4))
                continue
            line = line.lstrip("#* ").replace("**", "").replace("*", "")
            if line:
                elements.append(Paragraph(line, body_style))
        elements.append(Spacer(1, 12))

    # ── Doctor Notes ──
    if prediction.doctor_notes:
        elements.append(Paragraph("Doctor Notes", section_style))
        elements.append(Paragraph(prediction.doctor_notes, body_style))
        elements.append(Spacer(1, 12))

    # ── Disclaimer ──
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#E5E5E5"), spaceAfter=8))
    disclaimer = (
        "<b>IMPORTANT DISCLAIMER:</b> This report is generated by BreastGuard AI, an artificial "
        "intelligence screening tool based on deep learning. It is NOT a definitive medical diagnosis. "
        "Results should be reviewed and interpreted by a qualified medical professional. "
        "Always consult a licensed radiologist or oncologist for clinical decisions."
    )
    elements.append(Paragraph(disclaimer, ParagraphStyle(
        "Disclaimer", parent=body_style, fontSize=8, textColor=colors.HexColor("#7F8C8D"),
        borderColor=colors.HexColor("#E5E5E5"), borderWidth=1, borderPadding=8,
        backColor=LIGHT_GRAY
    )))

    doc.build(elements)
    buf.seek(0)
    return buf
