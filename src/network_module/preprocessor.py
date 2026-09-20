"""
Network Module — Preprocessor
Orders a network flow's features into the fixed vector the model expects.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class NetworkPreprocessor:
    """
    Builds the numeric feature vector for the Isolation Forest.
    The flow features are supplied by the caller (from a flow exporter or the
    dataset), so this orders and cleans them rather than computing them.
    """

    def __init__(self, feature_names: list[str]):
        self.feature_names = feature_names

    def build_feature_vector(self, features) -> np.ndarray:
        """
        Turn a flow into a (1, n_features) array in the trained feature order.

        Args:
            features: dict of {feature name: value}, or a list of values in the
                      trained feature order

        Returns:
            Feature array of shape (1, n_features)
        """
        if isinstance(features, dict):
            # Dataset column names carry leading spaces; strip them to match.
            cleaned = {key.strip(): value for key, value in features.items()}
            missing = [name for name in self.feature_names if name not in cleaned]
            if missing:
                raise ValueError(
                    f"Missing {len(missing)} feature(s), e.g. {missing[:3]}"
                )
            values = [cleaned[name] for name in self.feature_names]

        elif isinstance(features, list):
            if len(features) != len(self.feature_names):
                raise ValueError(
                    f"Expected {len(self.feature_names)} features, "
                    f"got {len(features)}"
                )
            values = features

        else:
            raise ValueError("features must be a dict or a list")

        # CICIDS2017 ratio features contain inf/NaN; replace to match training.
        vector = np.array([values], dtype=float)
        vector[~np.isfinite(vector)] = 0.0
        return vector
