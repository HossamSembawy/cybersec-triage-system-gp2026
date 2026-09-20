import pytest

from src.url_module.preprocessor import (
    FEATURE_NAMES,
    URLPreprocessor,
    extract_url_features,
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
