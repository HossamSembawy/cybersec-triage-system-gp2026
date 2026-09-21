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

URL_MODEL_PATH = os.getenv(
    "URL_MODEL_PATH", "models/url_module"
)

NETWORK_MODEL_PATH = os.getenv(
    "NETWORK_MODEL_PATH", "models/network_module"
)

DEFAULT_ORCHESTRATION_MODEL = "claude-sonnet-5"


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


@lru_cache(maxsize=1)
def _load_url_predictor():
    """
    Load the trained URLPredictor from disk.
    Called once at startup, result is cached for all requests.
    """
    from src.url_module.predictor import URLPredictor

    model_path = Path(URL_MODEL_PATH)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at: {model_path}\n"
            f"Place url_model.joblib in models/url_module/"
        )

    logger.info("Loading trained URL model from %s", model_path)
    return URLPredictor.from_pretrained(model_path)


def get_url_predictor():
    """FastAPI dependency — returns the cached URLPredictor."""
    return _load_url_predictor()


@lru_cache(maxsize=1)
def _load_network_predictor():
    """
    Load the trained NetworkPredictor from disk.
    Called once at startup, result is cached for all requests.
    """
    from src.network_module.predictor import NetworkPredictor

    model_path = Path(NETWORK_MODEL_PATH)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at: {model_path}\n"
            f"Place network_model.joblib in models/network_module/"
        )

    logger.info("Loading trained network model from %s", model_path)
    return NetworkPredictor.from_pretrained(model_path)


def get_network_predictor():
    """FastAPI dependency — returns the cached NetworkPredictor."""
    return _load_network_predictor()


@lru_cache(maxsize=1)
def _load_orchestration_service():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is required for orchestration"
        )

    from anthropic import Anthropic

    from src.orchestration.service import OrchestrationService

    model_version = os.getenv(
        "ANTHROPIC_MODEL",
        DEFAULT_ORCHESTRATION_MODEL,
    )
    client = Anthropic(api_key=api_key)

    logger.info("Creating orchestration service with %s", model_version)
    return OrchestrationService(
        client=client,
        model_version=model_version,
    )


def get_orchestration_service():
    """Return the cached orchestration service."""
    return _load_orchestration_service()
