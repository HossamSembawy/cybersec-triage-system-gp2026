"""
API Dependencies
Loads the fine-tuned DistilBERT email predictor once at startup.
The model is cached using lru_cache so weights are never reloaded
between requests.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

EMAIL_MODEL_PATH = os.getenv(
    "EMAIL_MODEL_PATH", "models/email_module"
)


@lru_cache(maxsize=1)
def _load_email_predictor():
    """
    Load the fine-tuned EmailPredictor from disk.
    Called once at startup, result is cached for all requests.
    """
    from src.email_module.predictor import EmailPredictor

    model_path = Path(EMAIL_MODEL_PATH)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at: {model_path}\n"
            f"Extract model files into models/email_module/"
        )

    logger.info("Loading fine-tuned model from %s", model_path)
    return EmailPredictor.from_pretrained(model_path)


def get_email_predictor():
    """FastAPI dependency — returns the cached EmailPredictor."""
    return _load_email_predictor()