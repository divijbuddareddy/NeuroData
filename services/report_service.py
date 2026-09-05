import os
from pathlib import Path
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

def generate_pdf_report(dataset_dict, output_pdf_path):
    """
    Generates a formal, multi-page publication-grade PDF Quality Control & Readiness Audit Report.
    """
    doc = SimpleDocTemplate(
        str(output_pdf_path),
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#0f172a')
    )
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#64748b')
    )
    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=16,
        textColor=colors.HexColor('#1e293b'),
        spaceBefore=12,
        spaceAfter=6
    )
    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#334155')
    )
    callout_style = ParagraphStyle(
        'Callout',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#1e40af')
    )

    story = []
    
    # Header Banner
    story.append(Paragraph("NeuroData Quality AI — Dataset Quality & Readiness Audit", title_style))
    story.append(Paragraph(f"Dataset: <b>{dataset_dict.get('name')}</b> | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Target: Pre-ML Screening", subtitle_style))
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#3b82f6'), spaceAfter=15))
    
    # Executive Summary Card (Table)
    readiness_label = dataset_dict.get('readiness_label', 'Not Ready')
    readiness_score = dataset_dict.get('readiness_score', 0)
    
    label_color = colors.HexColor('#10b981') if readiness_label == 'Ready' else (colors.HexColor('#f59e0b') if readiness_label == 'Requires Cleaning' else colors.HexColor('#ef4444'))
    
    summary_data = [
        [
            Paragraph("<b>Overall Readiness Score:</b>", body_style),
            Paragraph(f"<b><font color='{label_color}'>{readiness_score} / 100 ({readiness_label})</font></b>", body_style),
            Paragraph("<b>Total Scans Evaluated:</b>", body_style),
            Paragraph(f"{dataset_dict.get('total_scans')}", body_style)
        ],
        [
            Paragraph("<b>Good / Acceptable Scans:</b>", body_style),
            Paragraph(f"{dataset_dict.get('good_count')} Good / {dataset_dict.get('acceptable_count')} Acceptable", body_style),
            Paragraph("<b>Poor / Unusable Scans:</b>", body_style),
            Paragraph(f"{dataset_dict.get('poor_count')} Poor / {dataset_dict.get('unusable_count')} Unusable", body_style)
        ],
        [
            Paragraph("<b>Mean SNR / CNR:</b>", body_style),
            Paragraph(f"{dataset_dict.get('mean_snr')} dB / {dataset_dict.get('mean_cnr')}", body_style),
            Paragraph("<b>Outliers / Metadata %:</b>", body_style),
            Paragraph(f"{dataset_dict.get('outlier_count')} scans ({dataset_dict.get('metadata_completeness_pct')}%)", body_style)
        ]
    ]
    
    summary_table = Table(summary_data, colWidths=[1.8*inch, 1.8*inch, 1.8*inch, 1.8*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8fafc')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#cbd5e1')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 15))
    
    # Gemini AI Synthesis
    story.append(Paragraph("1. AI Research-Readiness Evaluation & Synthesis (Google Gemini)", h2_style))
    summary_text = dataset_dict.get('gemini_summary') or "Automated ML quality assessment completed."
    story.append(Paragraph(summary_text, body_style))
    story.append(Spacer(1, 8))
    
    if dataset_dict.get('gemini_risks'):
        story.append(Paragraph("<b>Downstream Machine Learning Risk Analysis:</b>", body_style))
        for line in dataset_dict.get('gemini_risks').split('\n'):
            if line.strip():
                story.append(Paragraph(line, body_style))
        story.append(Spacer(1, 8))
        
    if dataset_dict.get('gemini_recommendations'):
        story.append(Paragraph("<b>Actionable Preprocessing & Harmonization Checklist:</b>", body_style))
        recs = dataset_dict.get('gemini_recommendations')
        if isinstance(recs, str):
            for line in recs.split('\n'):
                if line.strip():
                    story.append(Paragraph(f"• {line.strip('-• ')}", body_style))
        elif isinstance(recs, list):
            for item in recs:
                story.append(Paragraph(f"• {item}", body_style))
        story.append(Spacer(1, 12))

    # Detailed Scan Breakdown Table
    story.append(Paragraph("2. Scan-Level Quality & Artifact Audit", h2_style))
    scan_rows = [[
        Paragraph("<b>Filename</b>", body_style),
        Paragraph("<b>Class</b>", body_style),
        Paragraph("<b>Confidence</b>", body_style),
        Paragraph("<b>SNR</b>", body_style),
        Paragraph("<b>Blur</b>", body_style),
        Paragraph("<b>Motion</b>", body_style),
        Paragraph("<b>Anomaly</b>", body_style)
    ]]
    
    scans = dataset_dict.get('scans', [])
    for s in scans[:20]: # Display top 20 scans in summary table
        scan_rows.append([
            Paragraph(s.get('filename', ''), body_style),
            Paragraph(f"<b>{s.get('quality_label')}</b>", body_style),
            Paragraph(f"{round(s.get('confidence', 0)*100)}%", body_style),
            Paragraph(f"{s.get('snr')}", body_style),
            Paragraph(f"{s.get('blur_score')}", body_style),
            Paragraph(f"{s.get('motion_score')}", body_style),
            Paragraph("OUTLIER" if s.get('is_outlier') else "Normal", body_style)
        ])
        
    scan_table = Table(scan_rows, colWidths=[1.8*inch, 0.9*inch, 0.9*inch, 0.7*inch, 0.7*inch, 0.7*inch, 1.5*inch])
    scan_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f1f5f9')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(scan_table)
    story.append(Spacer(1, 15))
    
    # Disclaimer
    story.append(Paragraph("<b>Disclaimer:</b> NeuroData Quality AI is a research-support quality assurance prototype. It is not intended for primary clinical diagnosis or patient management. All predictions are generated independently by PyTorch and Scikit-learn models, with Gemini generating narrative synthesis.", callout_style))
    
    doc.build(story)
    return str(output_pdf_path)
