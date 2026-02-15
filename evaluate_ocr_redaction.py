# evaluate_ocr_and_redaction.py
# Evaluation script for OCR and PDF redaction pipeline
# Computes Character Error Rate (CER), OCR-PII Recall,
# Redaction Completeness Rate (RCR), Leakage Rate, and Throughput.
import time
import difflib
from typing import List, Dict

from pdf_utils import PDFProcessor
from PyPDF2 import PdfReader


# 1. CHARACTER ERROR RATE (CER)

def cer(reference: str, hypothesis: str) -> float:
    """
    Character Error Rate = edit_distance / len(reference)
    Uses Python's difflib to approximate Levenshtein distance.
    """
    matcher = difflib.SequenceMatcher(None, reference, hypothesis)
    # Edit distance = substitutions + insertions + deletions
    # Approx via: len(ref) + len(hyp) - 2 * LCS
    lcs = sum(triple.size for triple in matcher.get_matching_blocks())
    edits = len(reference) + len(hypothesis) - 2 * lcs
    if len(reference) == 0:
        return 0.0
    return edits / len(reference)


def evaluate_ocr_cer(pairs: List[Dict[str, str]]):
    """
    pairs: list of { "id": ..., "gt_text": ..., "ocr_text": ... }
    gt_text = ground-truth text
    ocr_text = text produced by your OCR pipeline
    """
    cer_values = []
    for p in pairs:
        c = cer(p["gt_text"], p["ocr_text"])
        cer_values.append(c)
        print(f"[{p['id']}] CER = {c:.4f}")

    if cer_values:
        avg_cer = sum(cer_values) / len(cer_values)
        print(f"\nAverage CER over {len(cer_values)} samples: {avg_cer:.4f}")


# 2. OCR-PII RECALL

def ocr_pii_recall(ground_truth_pii: List[str], ocr_text: str) -> float:
    """
    Simple version: a PII token is 'recognized' if its exact string
    appears in the OCR text.
    """
    if not ground_truth_pii:
        return 0.0
    recognized = 0
    for token in ground_truth_pii:
        if token in ocr_text:
            recognized += 1
    return recognized / len(ground_truth_pii)


def demo_ocr_pii_recall():
    """
    Example: fill in with one of your scanned pages.
    """
    gt_text = "Dear John Doe, my SSN is 123-45-6789 and email is john@example.com."
    ocr_text = "Dear John Doe, my SSN is 123-45-678g and email is john@example.com"

    ground_truth_pii = ["John Doe", "123-45-6789", "john@example.com"]
    recall = ocr_pii_recall(ground_truth_pii, ocr_text)
    print(f"OCR-PII Recall = {recall:.4f}")


# 3. REDACTION COMPLETENESS + LEAKAGE

def extract_text_from_pdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages)


def redaction_completeness(original_text: str,
                           redacted_text: str,
                           pii_strings: List[str]) -> float:
    """
    RCR = correctly_removed / total
    A PII string is 'correctly removed' if it does NOT appear in redacted_text.
    """
    if not pii_strings:
        return 0.0
    removed = 0
    for pii in pii_strings:
        if pii not in redacted_text:
            removed += 1
    return removed / len(pii_strings)


def demo_redaction_metrics():
    """
    Example using a single PDF. You would prepare:
      - original_text (from original PDF)
      - redacted_text (from output PDF)
      - list of PII strings you know are in original
    """

    original_pdf = "data/pdf/sample_demo.pdf"
    redacted_pdf = "data/pdf_redacted/sample_demo_redacted.pdf"

    orig_text = extract_text_from_pdf(original_pdf)
    red_text = extract_text_from_pdf(redacted_pdf)

    # TODO: fill this list with real PII strings from your test doc
    pii_strings = [
        "John Doe",
        "123-45-6789",
        "john.doe@example.com",
        "123 Main St"
    ]

    rcr = redaction_completeness(orig_text, red_text, pii_strings)
    leakage = 1.0 - rcr
    print(f"Redaction Completeness Rate (RCR): {rcr:.4f}")
    print(f"Leakage Rate:                     {leakage:.4f}")


#  4. THROUGHPUT (PAGES PER MINUTE)

def evaluate_throughput(pdf_paths: List[str]):
    """
    Uses your PDFProcessor.redact_pdf to measure real runtime
    and computes pages/minute.
    """
    processor = PDFProcessor()
    total_pages = 0
    start = time.perf_counter()

    for pdf_path in pdf_paths:
        out_path = pdf_path.replace(".pdf", "_tmp_redacted.pdf")
        # Count pages
        reader = PdfReader(pdf_path)
        num_pages = len(reader.pages)
        total_pages += num_pages

        processor.redact_pdf(pdf_path, out_path)

    end = time.perf_counter()
    elapsed = end - start

    if elapsed == 0:
        elapsed = 1e-9

    pages_per_min = (total_pages / elapsed) * 60.0
    print(f"Processed {total_pages} pages in {elapsed:.2f} s")
    print(f"Throughput: {pages_per_min:.2f} pages/minute")


if __name__ == "__main__":
    print("=== DEMO: OCR-PII Recall ===")
    demo_ocr_pii_recall()

    print("\n=== DEMO: Redaction Completeness ===")
    demo_redaction_metrics()

    print("\n=== DEMO: Throughput ===")
    sample_pdfs = [
        "data/pdf/sample_demo.pdf",
        "data/pdf/sample_doc.pdf",
    ]
    evaluate_throughput(sample_pdfs)
