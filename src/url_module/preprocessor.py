"""
URL Module — Preprocessor
Lexical feature extraction from raw URL strings for Random Forest inference.
"""

from __future__ import annotations

import logging
import re
from urllib.parse import urlparse

import numpy as np

logger = logging.getLogger(__name__)

# Column order is fixed — the model was trained on features in exactly this
# order, so inference must produce them the same way.
FEATURE_NAMES = [
    "url_length",
    "hostname_length",
    "path_length",
    "count_dots",
    "count_hyphens",
    "count_at",
    "count_question",
    "count_equals",
    "count_percent",
    "count_slash",
    "count_digits",
    "subdomain_depth",
    "has_ip",
    "has_https",
    "has_port",
]

# Matches a bare dotted-quad host such as 192.168.0.1
IP_PATTERN = re.compile(r"^(\d{1,3}\.){3}\d{1,3}$")


def extract_url_features(url: str) -> list[float]:
    """
    Turn a raw URL string into the fixed lexical feature vector.

    This exact function is copied into the training notebook so the features
    used at training and inference match.

    Args:
        url: Raw URL string

    Returns:
        List of floats in FEATURE_NAMES order
    """
    url = url.strip()

    # Many dataset URLs omit the scheme, so add one for parsing only.
    has_scheme = "://" in url
    scheme = url.split("://", 1)[0].lower() if has_scheme else ""
    parse_target = url if has_scheme else "http://" + url

    # Some dataset URLs are malformed (e.g. unbalanced brackets) and cannot be
    # parsed; fall back to empty host/path so the lexical counts still run.
    try:
        parsed = urlparse(parse_target)
        hostname = parsed.hostname or ""
        path = parsed.path or ""
        has_port = parsed.port is not None
    except ValueError:
        hostname = ""
        path = ""
        has_port = False

    return [
        float(len(url)),
        float(len(hostname)),
        float(len(path)),
        float(url.count(".")),
        float(url.count("-")),
        float(url.count("@")),
        float(url.count("?")),
        float(url.count("=")),
        float(url.count("%")),
        float(url.count("/")),
        float(sum(character.isdigit() for character in url)),
        float(hostname.count(".")),
        float(1 if IP_PATTERN.match(hostname) else 0),
        float(1 if scheme == "https" else 0),
        float(1 if has_port else 0),
    ]


class URLPreprocessor:
    """
    Builds the lexical feature vector expected by the Random Forest classifier.
    """

    def extract_features(self, url: str) -> np.ndarray:
        """
        Extract features from a single URL for predict_proba.

        Args:
            url: Raw URL string

        Returns:
            Feature array of shape (1, n_features)
        """
        if not url or not url.strip():
            raise ValueError("Input URL must not be empty")

        features = extract_url_features(url)
        return np.array([features], dtype=float)
