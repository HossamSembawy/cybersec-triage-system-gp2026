import joblib
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.url_module.predictor import URLPredictor
from src.url_module.preprocessor import (
    FEATURE_NAMES,
    URLPreprocessor,
    extract_url_features,
)

LABEL_NAMES = ["benign", "defacement", "phishing", "malware"]


def save_synthetic_bundle(directory, feature_names):
    random_generator = np.random.default_rng(0)
    features = random_generator.random((40, len(FEATURE_NAMES)))
    labels = np.arange(40) % len(LABEL_NAMES)
    model = RandomForestClassifier(n_estimators=5, random_state=0)
    model.fit(features, labels)
    joblib.dump(
        {
            "model": model,
            "feature_names": feature_names,
            "label_names": LABEL_NAMES,
            "model_version": "test",
        },
        directory / "url_model.joblib",
    )


def test_extract_url_features_returns_documented_order():
    url = "https://sub.example.com:8443/path-1?q=2"
    expected_by_name = {
        "url_length": float(len(url)),
        "hostname_length": 15.0,
        "path_length": 7.0,
        "count_dots": 2.0,
        "count_hyphens": 1.0,
        "count_at": 0.0,
        "count_question": 1.0,
        "count_equals": 1.0,
        "count_percent": 0.0,
        "count_slash": 3.0,
        "count_digits": 6.0,
        "subdomain_depth": 2.0,
        "has_ip": 0.0,
        "has_https": 1.0,
        "has_port": 1.0,
    }

    features = extract_url_features(url)

    assert len(FEATURE_NAMES) == 15
    assert features == [expected_by_name[name] for name in FEATURE_NAMES]


def test_malformed_url_does_not_raise():
    features = extract_url_features("http://[example.com/path")

    assert len(features) == len(FEATURE_NAMES)


@pytest.mark.parametrize("url", ["", "   ", "\t"])
def test_empty_url_is_rejected(url):
    preprocessor = URLPreprocessor()

    with pytest.raises(ValueError, match="must not be empty"):
        preprocessor.extract_features(url)


def test_bundle_with_matching_feature_order_loads(tmp_path):
    save_synthetic_bundle(tmp_path, list(FEATURE_NAMES))

    predictor = URLPredictor.from_pretrained(tmp_path)

    assert predictor.predict("https://example.com/login").severity in {
        "LOW",
        "MEDIUM",
        "HIGH",
    }


def test_bundle_with_reordered_features_is_rejected(tmp_path):
    reordered_names = list(reversed(FEATURE_NAMES))
    save_synthetic_bundle(tmp_path, reordered_names)

    with pytest.raises(ValueError, match="feature order does not match"):
        URLPredictor.from_pretrained(tmp_path)
