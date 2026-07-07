"""Generates the 20 synthetic test-invoice PDFs (+ a manifest.json of
expected outcomes) used to demonstrate the validation/matching/flagging
logic. Run:

    python test_invoices/generate_test_invoices.py

Uses reportlab with a monospace (Courier) font so the PDF's text layer
preserves column spacing exactly -- this keeps intake-node text
extraction (pdfplumber) faithful to the original layout regardless of
which downstream extractor (LLM or offline heuristic) reads it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from test_invoices.invoice_data import CASES, render_invoice_text

OUT_DIR = Path(__file__).resolve().parent / "invoices"


def render_pdf(text: str, path: Path) -> None:
    c = canvas.Canvas(str(path), pagesize=letter)
    c.setFont("Courier", 10)
    width, height = letter
    x, y = 50, height - 60
    for line in text.splitlines():
        c.drawString(x, y, line)
        y -= 14
        if y < 50:
            c.showPage()
            c.setFont("Courier", 10)
            y = height - 60
    c.save()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []

    for case in CASES:
        text = render_invoice_text(case)
        pdf_path = OUT_DIR / f"{case.name}.pdf"
        render_pdf(text, pdf_path)

        manifest.append(
            {
                "name": case.name,
                "category": case.category,
                "file": f"invoices/{case.name}.pdf",
                "expected_status": case.expected_status,
                "expected_reason_codes": case.expected_reason_codes,
                "notes": case.notes,
            }
        )
        print(f"wrote {pdf_path}")

    manifest_path = Path(__file__).resolve().parent / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {manifest_path} ({len(manifest)} cases)")


if __name__ == "__main__":
    main()
