"""
URL Module — Predictor
Loads the trained Random Forest bundle and returns structured predictions.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import joblib

from src.url_module.preprocessor import FEATURE_NAMES, URLPreprocessor

logger = logging.getLogger(__name__)

MODEL_VERSION = "url-randomforest-v1"

# Same thresholds as the email module so severity means the same thing
# across the whole system.
SEVERITY_THRESHOLDS = {
    "HIGH":   0.70,
    "MEDIUM": 0.40,
}

# The label that means "not a threat"; everything else is malicious.
BENIGN_LABEL = "benign"


@dataclass
class PredictionResult:
    """Structured output from the URL detection module."""
    prediction:            str
    malicious_probability: float
    severity:              str
    label:                 int
    model_version:         str
    inference_latency_ms:  float
    timestamp:             str
    class_probabilities:   dict = field(default_factory=dict)


def _classify_severity(malicious_probability: float) -> str:
    if malicious_probability >= SEVERITY_THRESHOLDS["HIGH"]:
        return "HIGH"
    if malicious_probability >= SEVERITY_THRESHOLDS["MEDIUM"]:
        return "MEDIUM"
    return "LOW"


class URLPredictor:
    """
    Inference wrapper for the trained Random Forest URL classifier.

    Usage:
        predictor = URLPredictor.from_pretrained("models/url_module")
        result = predictor.predict("http://paypa1-secure-login.com/verify")
    """

    def __init__(self, model, label_names, preprocessor, model_version=MODEL_VERSION):
        self.model = model
        self.label_names = label_names
        self.preprocessor = preprocessor
        self.model_version = model_version
        logger.info(
            "URLPredictor ready (%d classes, %s)",
            len(label_names),
            model_version,
        )

    @classmethod
    def from_pretrained(cls, model_path: str | Path) -> "URLPredictor":
        """
        Load the trained model bundle from disk.

        Args:
            model_path: Directory containing url_model.joblib

        Returns:
            Initialised URLPredictor
        """
        path = Path(model_path)
        bundle_file = path / "url_model.joblib"
        if not bundle_file.exists():
            raise FileNotFoundError(
                f"Model not found at: {bundle_file}\n"
                f"Place url_model.joblib in models/url_module/"
            )

        logger.info("Loading model from %s", bundle_file)
        bundle = joblib.load(bundle_file)

        # A reordered feature list would still load but silently mispredict.
        if list(bundle["feature_names"]) != FEATURE_NAMES:
            raise ValueError(
                "Saved URL feature order does not match the serving "
                f"preprocessor.\nSaved:   {list(bundle['feature_names'])}\n"
                f"Serving: {FEATURE_NAMES}"
            )

        return cls(
            model=bundle["model"],
            label_names=bundle["label_names"],
            preprocessor=URLPreprocessor(),
            model_version=bundle.get("model_version", MODEL_VERSION),
        )

    def predict(self, url: str) -> PredictionResult:
        """
        Run inference on a single URL.

        Args:
            url: Raw URL string

        Returns:
            PredictionResult with threat class, malicious probability, severity
        """
        if not url or not url.strip():
            raise ValueError("Input URL must not be empty")

        start = time.perf_counter()

        features = self.preprocessor.extract_features(url)
        probabilities = self.model.predict_proba(features)[0]

        # The model was trained on integer labels 0..n in label_names order,
        # so predict_proba column i corresponds to label_names[i].
        class_probabilities = {}
        for class_index, name in enumerate(self.label_names):
            class_probabilities[name] = float(probabilities[class_index])

        predicted_name = max(class_probabilities, key=class_probabilities.get)
        label = self.label_names.index(predicted_name)
        benign_probability = class_probabilities.get(BENIGN_LABEL, 0.0)
        malicious_probability = 1.0 - benign_probability
        latency_ms = (time.perf_counter() - start) * 1000

        return PredictionResult(
            prediction=predicted_name,
            malicious_probability=malicious_probability,
            severity=_classify_severity(malicious_probability),
            label=label,
            model_version=self.model_version,
            inference_latency_ms=latency_ms,
            timestamp=datetime.now(timezone.utc).isoformat(),
            class_probabilities=class_probabilities,
        )
