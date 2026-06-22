# JUDGELYTICS - FastAPI Backend: PDF Report Service
# Purpose: Generate professional PDF reports using ReportLab
# Authors: Rakhi Tiwari (EN23CS302834), Raghav Dubey (EN23CS301816)
# Date: January 2026

"""
PDF report generation service for Judgelytics backend.

Generates professional PDF reports with case analysis and filing guidance.
"""

import logging
from typing import Dict, Any, Tuple
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.pdfgen import canvas
from io import BytesIO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PDFService:
    """PDF report generation service."""

    @staticmethod
    def sanitize(text: str) -> str:
        """Sanitize text for ReportLab standard fonts."""
        if not text:
            return ""
        # ReportLab's standard Helvetica font does not support the Rupee symbol
        return str(text).replace('₹', 'Rs. ').replace('✓', '[Y]').replace('✗', '[N]')

    @staticmethod
    def generate_report_pdf(
        case_data: Dict[str, Any],
        prediction: Dict[str, Any],
        user_data: Dict[str, Any]
    ) -> bytes:
        """
        Generate professional PDF report.

        Args:
            case_data (dict): Case information
            prediction (dict): Prediction results
            user_data (dict): User information

        Returns:
            bytes: PDF content as bytes
        """

        try:
            logger.info("Generating PDF report...")

            # Create PDF buffer
            pdf_buffer = BytesIO()

            # Create PDF document
            doc = SimpleDocTemplate(
                pdf_buffer,
                pagesize=A4,
                topMargin=0.5 * inch,
                bottomMargin=0.5 * inch,
                leftMargin=0.75 * inch,
                rightMargin=0.75 * inch
            )

            # Container for PDF elements
            elements = []

            # Get styles
            styles = getSampleStyleSheet()

            # Custom styles
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                textColor=colors.HexColor('#1a3a52'),
                spaceAfter=6,
                alignment=1  # Center
            )
            
            subtitle_style = ParagraphStyle(
                'Subtitle', 
                parent=styles['Normal'], 
                fontSize=12, 
                textColor=colors.HexColor('#666666'), 
                spaceAfter=15, 
                alignment=1
            )

            heading_style = ParagraphStyle(
                'CustomHeading',
                parent=styles['Heading2'],
                fontSize=14,
                textColor=colors.HexColor('#2c5aa0'),
                spaceAfter=8,
                spaceBefore=12
            )
            
            normal_style = styles['Normal']
            normal_style.leading = 14

            # Header
            elements.append(Paragraph("<b>JUDGELYTICS</b>", title_style))
            elements.append(Paragraph("AI-Powered Legal Judgment Prediction Report", subtitle_style))

            # Meta Info (User & Date)
            meta_text = (
                f"<b>Prepared For:</b> {user_data.get('name', 'N/A')} ({user_data.get('email', 'N/A')})<br/>"
                f"<b>Case ID:</b> {case_data.get('case_id', 'N/A')} | "
                f"<b>Generated:</b> {datetime.now().strftime('%d %b %Y, %H:%M')}"
            )
            elements.append(Paragraph(meta_text, normal_style))
            elements.append(Spacer(1, 0.15 * inch))

            # Verdict Banner
            outcome = prediction.get('outcome', 'Unknown')
            win_prob_pct = prediction.get('win_probability_pct', 0)

            # Color based on outcome
            if outcome == "Allowed":
                verdict_color = colors.HexColor('#28a745')
                verdict_text = "LIKELY ALLOWED"
            elif outcome == "Partially Allowed":
                verdict_color = colors.HexColor('#ffc107')
                verdict_text = "PARTIALLY ALLOWED"
            else:
                verdict_color = colors.HexColor('#dc3545')
                verdict_text = "LIKELY DISMISSED"

            # Verdict box
            verdict_table = Table(
                [[Paragraph(f"<font color='white' size=16><b>{verdict_text} ({win_prob_pct}% Win Probability)</b></font>", ParagraphStyle('VT', alignment=1))]],
                colWidths=[6.5 * inch],
                rowHeights=[0.4 * inch]
            )
            verdict_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), verdict_color),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ]))
            elements.append(verdict_table)
            elements.append(Spacer(1, 0.2 * inch))

            # Case Summary Section
            elements.append(Paragraph("Case Summary", heading_style))
            
            description = PDFService.sanitize(case_data.get('description', 'No description provided.'))
            # Truncate description if it's too long for a single cell to avoid breaking the table
            if len(description) > 800:
                description = description[:800] + "..."
                
            summary_data = [
                ["Category / Sector", f"{case_data.get('category', 'N/A')} / {case_data.get('sector', 'N/A')}"],
                ["Claim Amount", PDFService.sanitize(str(case_data.get('claim_amount', 'N/A')))],
                ["Description", Paragraph(description, normal_style)]
            ]
            
            summary_table = Table(summary_data, colWidths=[1.8 * inch, 4.7 * inch])
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f8f9fa')),
                ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#495057')),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dee2e6')),
                ('PADDING', (0, 0), (-1, -1), 8),
            ]))
            elements.append(summary_table)
            elements.append(Spacer(1, 0.2 * inch))

            # Key Metrics Section
            elements.append(Paragraph("AI Analysis Metrics", heading_style))
            
            metrics_data = [
                ["Predicted Outcome", outcome, "Confidence Level", prediction.get('confidence', 'N/A').title()],
                ["Recommended Forum", PDFService.sanitize(prediction.get('recommended_forum', 'N/A')), "Evidence Strength", prediction.get('evidence_strength', 'N/A').title()],
                ["Est. Filing Fee", PDFService.sanitize(str(prediction.get('filing_fee', 'N/A'))), "Similar Cases", str(prediction.get('similar_cases_count', 'N/A'))],
            ]
            
            metrics_table = Table(metrics_data, colWidths=[1.6 * inch, 1.65 * inch, 1.6 * inch, 1.65 * inch])
            metrics_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e9ecef')),
                ('BACKGROUND', (2, 0), (2, -1), colors.HexColor('#e9ecef')),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTNAME', (2, 0), (2, -1), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dee2e6')),
                ('PADDING', (0, 0), (-1, -1), 6),
            ]))
            elements.append(metrics_table)
            elements.append(Spacer(1, 0.2 * inch))

            # Applicable Laws
            elements.append(Paragraph("Applicable Legal Sections", heading_style))
            sections = prediction.get('applicable_sections', [])
            section_text = " • ".join([PDFService.sanitize(s) for s in sections]) if sections else "Not determined"
            elements.append(Paragraph(section_text, normal_style))
            elements.append(Spacer(1, 0.2 * inch))

            # Step-by-Step Filing Process
            elements.append(Paragraph("Step-by-Step Filing Process", heading_style))
            filing_steps = [
                "1. <b>Prepare Complaint:</b> Draft complaint with all relevant facts and evidence.",
                "2. <b>Gather Documents:</b> Collect purchase receipts, warranty cards, correspondence.",
                "3. <b>Calculate Relief:</b> Document all financial losses and mental agony.",
                "4. <b>Submit Application:</b> File complaint with Consumer Commission with applicable fee.",
                "5. <b>Attend Hearings:</b> Present your case and evidence before the Commission.",
                "6. <b>Receive Order:</b> Obtain certified copy of judgment order."
            ]
            
            for step in filing_steps:
                elements.append(Paragraph(step, normal_style))
                elements.append(Spacer(1, 0.05 * inch))

            # Footer
            elements.append(Spacer(1, 0.4 * inch))
            footer_text = (
                f"<font size=8 color='grey'><i>This report is for informational purposes only and does not constitute legal advice. "
                f"Please consult with a qualified attorney. | Report generated by Judgelytics AI Platform.</i></font>"
            )
            elements.append(Paragraph(footer_text, normal_style))

            # Build PDF
            doc.build(elements)

            # Get PDF bytes
            pdf_bytes = pdf_buffer.getvalue()
            pdf_buffer.close()

            logger.info(f"PDF report generated: {len(pdf_bytes)} bytes")

            return pdf_bytes

        except Exception as e:
            logger.error(f"Failed to generate PDF: {str(e)}")
            raise


# Create global PDF service instance
pdf_service = PDFService()


def get_pdf_service() -> PDFService:
    """
    Get PDF service instance.

    Returns:
        PDFService: PDF service instance
    """
    return pdf_service
