"""Node 1: Intake -- OCR / text extraction from the source document.

Native-text PDFs (the common case for vendor-generated invoices) are read
directly via pdfplumber. Scanned PDFs and image files (png/jpg/tiff) are
rasterized and OCR'd with pytesseract, i.e. a genuine OCR pass, not just
a text-layer read.
"""
from __future__ import annotations

from pathlib import Path

from agent.state import InvoiceState


def _extract_pdf_text(path: Path) -> str:
    import pdfplumber

    with pdfplumber.open(path) as pdf:
        text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    return text.strip()


def _ocr_pdf(path: Path) -> str:
    from pdf2image import convert_from_path
    import pytesseract

    images = convert_from_path(str(path))
    return "\n".join(pytesseract.image_to_string(img) for img in images).strip()


def _ocr_image(path: Path) -> str:
    import pytesseract
    from PIL import Image

    with Image.open(path) as img:
        return pytesseract.image_to_string(img).strip()


async def intake_node(state: InvoiceState) -> dict:
    path = Path(state["file_path"])
    audit_log = list(state.get("audit_log", []))

    if not path.exists():
        raise FileNotFoundError(f"Invoice file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        text = _extract_pdf_text(path)
        if len(text) < 20:
            text = _ocr_pdf(path)
            audit_log.append("intake: PDF had no text layer, ran OCR via pytesseract")
            ocr_confidence = 0.75
        else:
            audit_log.append("intake: extracted native PDF text layer via pdfplumber")
            ocr_confidence = 0.99
    elif suffix in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
        text = _ocr_image(path)
        audit_log.append("intake: ran OCR via pytesseract on image")
        ocr_confidence = 0.75
    elif suffix == ".txt":
        text = path.read_text(encoding="utf-8").strip()
        audit_log.append("intake: read plain text file directly")
        ocr_confidence = 1.0
    else:
        raise ValueError(f"Unsupported invoice file type: {suffix}")

    return {
        "raw_text": text,
        "ocr_lines": text.splitlines(),
        "ocr_confidence": ocr_confidence,
        "audit_log": audit_log,
    }
