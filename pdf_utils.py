# pdf_utils.py
# PDF processing utilities for AI-based PII redaction
# Supports text extraction (text-based and OCR) and redaction
# PDF creation from text

import os
import textwrap
import tempfile

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import LETTER
from PyPDF2 import PdfReader

from redactor import Redactor

# The following are imports for OCR
from pdf2image import convert_from_path
import pytesseract
from PIL import Image

PDF_DIR = os.path.join("data", "pdf")
PDF_REDACTED_DIR = os.path.join("data", "pdf_redacted")

class PDFProcessor:
    """
    Handles PDF I/O + redaction using the Redactor class.
    Supports:
      - Text-based PDFs via PyPDF2
      - Scanned PDFs via OCR fallback (pdf2image + pytesseract)
    """

    def __init__(self):
        self.redactor = Redactor()

    def text_to_pdf(self, text: str, pdf_path: str, max_chars: int = 90):
        os.makedirs(os.path.dirname(pdf_path), exist_ok=True)

        c = canvas.Canvas(pdf_path, pagesize=LETTER)
        width, height = LETTER
        y = height - 72

        lines = []
        for line in text.split("\n"):
            wrapped = textwrap.wrap(line, max_chars) or [""]
            lines.extend(wrapped)

        for line in lines:
            if y < 72:
                c.showPage()
                y = height - 72
            c.drawString(72, y, line)
            y -= 14

        c.save()

    def _pdf_to_text_simple(self, pdf_path: str) -> str:
        """
        Try to extract text using PyPDF2 (works for text PDFs).
        """
        reader = PdfReader(pdf_path)
        pages = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        return "\n".join(pages).strip()

    def _pdf_to_text_ocr(self, pdf_path: str) -> str:
        """
        OCR-based extraction for scanned PDFs.
        Uses pdf2image + pytesseract.
        """
        print("🔹 No text extracted. Falling back to OCR...")
        pages_text = []

        # Convert pages to images (use a temp folder)
        with tempfile.TemporaryDirectory() as tmpdir:
            images = convert_from_path(pdf_path, output_folder=tmpdir)
            for idx, img in enumerate(images):
                if not isinstance(img, Image.Image):
                    img = img.convert("RGB")
                page_text = pytesseract.image_to_string(img)
                pages_text.append(page_text)

        return "\n".join(pages_text).strip()

    def pdf_to_text(self, pdf_path: str) -> str:
        """
        Smart text extractor:
          1) Try normal extraction
          2) If empty -> OCR
        """
        text = self._pdf_to_text_simple(pdf_path)
        if text:
            print("🔹 Extracted text using PyPDF2 (text-based PDF).")
            return text

        # Fallback to OCR
        text = self._pdf_to_text_ocr(pdf_path)
        print("🔹 Extracted text using OCR.")
        return text

    def redact_pdf(self, input_pdf: str, output_pdf: str):
        """
        End-to-end PDF redaction:
          1) Extract text (text-based or OCR)
          2) Redact PII
          3) Create a new PDF from redacted text
        """
        print(f" Reading PDF: {input_pdf}")
        text = self.pdf_to_text(input_pdf)

        print(" Redacting text...")
        redacted_text = self.redactor.redact_text(text)

        print(f" Writing redacted PDF: {output_pdf}")
        self.text_to_pdf(redacted_text, output_pdf)
        print(" Redacted PDF created.")
