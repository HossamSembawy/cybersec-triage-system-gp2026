"""
Network Module — Predictor
Loads the trained Isolation Forest bundle and returns structured predictions.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib

from src.network_module.preprocessor import NetworkPreprocessor

logger = logging.getLogger(__name__)

MODEL_VERSION = "network-isolationforest-v1"

# Same thresholds as the email and URL modules so severity means the same
# thing across the whole system.
SEVERITY_THRESHOLDS = {
    "HIGH":   0.70,
    "MEDIUM": 0.40,
}


@dataclass
class PredictionResult:
    """Structured output from the network anomaly module."""
    prediction:           str
    anomaly_probability:  float
    severity:             str
    is_anomaly:           bool
    model_version:        str
    inference_latency_ms: float
    timestamp:            str


def _classify_severity(anomaly_probability: float) -> str:
    if anomaly_probability >= SEVERITY_THRESHOLDS["HIGH"]:
        return "HIGH"
    if anomaly_probability >= SEVERITY_THRESHOLDS["MEDIUM"]:
        return "MEDIUM"
    return "LOW"


class NetworkPredictor:
    """
    Inference wrapper for the trained Isolation Forest anomaly detector.

    Usage:
        predictor = NetworkPredictor.from_pretrained("models/network_module")
        result = predictor.predict({"Flow Duration": 112, ...})
    """

    def __init__(self, model, feature_names, score_lo, score_hi,
                 preprocessor, model_version=MODEL_VERSION):
        self.model = model
        self.feature_names = feature_names
        self.score_lo = score_lo
        self.score_hi = score_hi
        self.preprocessor = preprocessor
        self.model_version = model_version
        logger.info(
            "NetworkPredictor ready (%d features, %s)",
            len(feature_names), model_version
        )

    @classmethod
    def from_pretrained(cls, model_path: str | Path) -> "NetworkPredictor":
        """
        Load the trained model bundle from disk.

        Args:
            model_path: Directory containing network_model.joblib

        Returns:
            Initialised NetworkPredictor
        """
        path = Path(model_path)
        bundle_file = path / "network_model.joblib"
        if not bundle_file.exists():
            raise FileNotFoundError(
                f"Model not found at: {bundle_file}\n"
                f"Place network_model.joblib in models/network_module/"
            )

        logger.info("Loading model from %s", bundle_file)
        bundle = joblib.load(bundle_file)
        feature_names = bundle["feature_names"]

        return cls(
            model=bundle["model"],
            feature_names=feature_names,
            score_lo=bundle["score_lo"],
            score_hi=bundle["score_hi"],
            preprocessor=NetworkPreprocessor(feature_names),
            model_version=bundle.get("model_version", MODEL_VERSION),
        )

    def predict(self, features) -> PredictionResult:
        """
        Run anomaly detection on a single flow.

        Args:
            features: dict of {feature name: value} or an ordered list of values

        Returns:
            PredictionResult with anomaly probability, severity, and flag
        """
        start = time.perf_counter()

        vector = self.preprocessor.build_feature_vector(features)

        # Higher raw score = more anomalous. Normalise to [0, 1] using the
        # percentile bounds learned from benign training traffic: the benign
        # median maps to 0 (normal) and the benign upper tail maps to 1.
        raw_score = -self.model.score_samples(vector)[0]
        score_range = self.score_hi - self.score_lo
        if score_range == 0:
            anomaly_probability = 0.0
        else:
            anomaly_probability = (raw_score - self.score_lo) / score_range
        anomaly_probability = float(min(max(anomaly_probability, 0.0), 1.0))

        severity = _classify_severity(anomaly_probability)
        is_anomaly = anomaly_probability >= SEVERITY_THRESHOLDS["MEDIUM"]
        latency_ms = (time.perf_counter() - start) * 1000

        return PredictionResult(
            prediction="anomaly" if is_anomaly else "normal",
            anomaly_probability=anomaly_probability,
            severity=severity,
            is_anomaly=is_anomaly,
            model_version=self.model_version,
            inference_latency_ms=latency_ms,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
