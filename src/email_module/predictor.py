"""
Email Module — Predictor
Loads fine-tuned DistilBERT and returns structured predictions.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import DistilBertForSequenceClassification

from src.email_module.preprocessor import EmailPreprocessor

logger = logging.getLogger(__name__)

MODEL_VERSION = "distilbert-phishing-v1"

SEVERITY_THRESHOLDS = {
    "HIGH":   0.70,
    "MEDIUM": 0.40,
}

LABELS = {0: "legitimate", 1: "phishing"}


@dataclass
class PredictionResult:
    """Structured output from the email detection module."""
    prediction:            str
    phishing_probability:  float
    severity:              str
    label:                 int
    model_version:         str
    inference_latency_ms:  float
    timestamp:             str
    top_tokens:            list = field(default_factory=list)


def _classify_severity(probability: float) -> str:
    if probability >= SEVERITY_THRESHOLDS["HIGH"]:
        return "HIGH"
    if probability >= SEVERITY_THRESHOLDS["MEDIUM"]:
        return "MEDIUM"
    return "LOW"


class EmailPredictor:
    """
    Inference wrapper for the fine-tuned DistilBERT classifier.

    Usage:
        predictor = EmailPredictor.from_pretrained("models/email_module")
        result = predictor.predict("urgent verify your account...")
    """

    def __init__(self, model, preprocessor, device=None):
        self.model = model
        self.preprocessor = preprocessor
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        self.model.eval()
        logger.info("EmailPredictor ready on device: %s", self.device)

    @classmethod
    def from_pretrained(cls, model_path: str | Path) -> "EmailPredictor":
        """
        Load fine-tuned model from disk.

        Args:
            model_path: Directory containing model weights

        Returns:
            Initialised EmailPredictor
        """
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Model not found at: {path}\n"
                f"Place model files in models/email_module/"
            )

        logger.info("Loading model from %s", path)
        model = DistilBertForSequenceClassification.from_pretrained(
            str(path),
            use_safetensors=True,
        )
        preprocessor = EmailPreprocessor()
        return cls(model=model, preprocessor=preprocessor)

    def predict(self, text: str) -> PredictionResult:
        """
        Run inference on a single email.

        Args:
            text: Email text in text_combined format

        Returns:
            PredictionResult with prediction, probability, severity
        """
        if not text or not text.strip():
            raise ValueError("Input text must not be empty")

        start = time.perf_counter()

        encoding = self.preprocessor.tokenize_single(text)
        input_ids      = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            probabilities = torch.softmax(outputs.logits, dim=-1)

        phishing_prob = probabilities[0][1].item()
        label         = int(phishing_prob >= 0.5)
        latency_ms    = (time.perf_counter() - start) * 1000

        return PredictionResult(
            prediction=LABELS[label],
            phishing_probability=phishing_prob,
            severity=_classify_severity(phishing_prob),
            label=label,
            model_version=MODEL_VERSION,
            inference_latency_ms=latency_ms,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def predict_from_fields(
        self,
        sender:  str = "",
        date:    str = "",
        subject: str = "",
        body:    str = "",
    ) -> PredictionResult:
        """
        Predict from structured email fields.
        Concatenates into text_combined format before inference.
        """
        text = EmailPreprocessor.build_text_combined(
            sender, date, subject, body
        )
        return self.predict(text)