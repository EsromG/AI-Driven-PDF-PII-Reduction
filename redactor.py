# redactor.py
# Hybrid PII redactor:
# - Regex for obvious patterns (email, phone, SSN, address, IDs)
# - Transformer NER model for contextual entities (names, addresses, etc.)  

import os
import re
import torch

from transformers import AutoTokenizer, AutoModelForTokenClassification

# Path to the trained model directory (from train_ner.py)
MODEL_DIR = os.path.join("models", "pii_ner")


# Regex rules for hard PII

# Email: simple RFC-like pattern
EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")

# Phone: allow +, spaces, dashes, parentheses, at least 7 digits total
PHONE_RE = re.compile(r"\+?\d[\d\s\-\(\)]{7,}\d")

# SSN: 123-45-6789 style
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

# 9-digit IDs: loose fallback (student IDs, account IDs, etc.)
NINE_DIGIT_RE = re.compile(r"\b\d{9}\b")

# Address-like patterns

# 1) Street with suffix: "123 Main St", "55 Oak Road", etc.
ADDRESS_STREET_RE = re.compile(
    r"\b\d{1,6}\s+[A-Za-z0-9]+\s+(Street|St|Road|Rd|Avenue|Ave|Lane|Ln|Boulevard|Blvd|Court|Ct)\b",
    re.IGNORECASE,
)

# 2) "County"
COUNTY_RE = re.compile(
    r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\s+County\b"
)

CITY_STATE_ZIP_RE = re.compile(
    r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*,\s?[A-Z]{2}\s?\d{5}\b"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Redactor:
    """
    Hybrid PII redactor:
    - Regex for obvious patterns (email, phone, SSN, address, IDs)
    - Transformer NER model for contextual entities (names, addresses, etc.)
    """

    def __init__(self):
        print("🔹 Loading redaction model from:", MODEL_DIR)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        self.model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR).to(DEVICE)
        self.id2label = self.model.config.id2label

    def _apply_regex_rules(self, text: str) -> str:
        """
        First pass: rule-based redaction.

        IMPORTANT: order matters.
        We redact SSN BEFORE phone so SSN doesn't get tagged as [PHONE].
        """
        # Email
        text = EMAIL_RE.sub("[EMAIL]", text)

        # SSN before phone (so 123-45-6789 becomes [SSN], not [PHONE])
        text = SSN_RE.sub("[SSN]", text)

        # Phone
        text = PHONE_RE.sub("[PHONE]", text)

        # Address-like patterns
        text = ADDRESS_STREET_RE.sub("[ADDRESS]", text)
        text = COUNTY_RE.sub("[ADDRESS]", text)
        text = CITY_STATE_ZIP_RE.sub("[ADDRESS]", text)


        # Loose 9-digit IDs
        text = NINE_DIGIT_RE.sub("[ID]", text)

        return text

    def _pii_spans_from_model(self, text: str, threshold: float = 0.3):
        """
        Use the trained NER model to get PII spans with character offsets.
        Lower threshold => more sensitive to possible PII.
        """
        encoding = self.tokenizer(
            text,
            return_offsets_mapping=True,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        )
        offsets = encoding.pop("offset_mapping")[0].tolist()
        encoding = {k: v.to(DEVICE) for k, v in encoding.items()}

        with torch.no_grad():
            logits = self.model(**encoding).logits[0]
            probs = torch.softmax(logits, dim=-1)
            scores, preds = torch.max(probs, dim=-1)

        spans = []
        current = None

        for (start, end), label_id, score in zip(offsets, preds.tolist(), scores.tolist()):
            # Skip special tokens ([CLS], [SEP]) which typically have (0, 0)
            if start == 0 and end == 0:
                continue

            label_name = self.id2label[label_id]  # e.g. "B-FIRSTNAME" or "O"

            if label_name == "O" or score < threshold:
                if current:
                    spans.append(current)
                    current = None
                continue

            prefix, ent_type = label_name.split("-", 1)

            if prefix == "B" or current is None or current["label"] != ent_type:
                if current:
                    spans.append(current)
                current = {"start": start, "end": end, "label": ent_type}
            else:
                # I- tag continuing the same entity
                current["end"] = end

        if current:
            spans.append(current)

        return spans

    def redact_text(self, text: str, use_rules: bool = True, threshold: float = 0.3) -> str:
        """
        Full redaction pipeline on plain text.
        - use_rules: apply regex layer first
        - threshold: probability cutoff for model predictions
        """
        if use_rules:
            text = self._apply_regex_rules(text)

        spans = self._pii_spans_from_model(text, threshold=threshold)
        chars = list(text)

        # Replace from end to start so indices don’t shift
        for span in sorted(spans, key=lambda s: s["start"], reverse=True):
            start, end = span["start"], span["end"]
            length = max(end - start, 1)
            chars[start:end] = list("█" * length)

        return "".join(chars)

    def debug_spans(self, text: str, threshold: float = 0.3):
        """
        Print model-detected spans for debugging (no masking).
        """
        spans = self._pii_spans_from_model(text, threshold=threshold)
        print("🔍 Model spans:")
        if not spans:
            print("  (no spans detected)")
        for s in spans:
            snippet = text[s["start"]:s["end"]]
            print(f"  [{s['label']}] '{snippet}'  ({s['start']}–{s['end']})")
