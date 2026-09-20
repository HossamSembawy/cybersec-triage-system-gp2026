import pytest

from src.email_module.predictor import _classify_severity as email_severity
from src.network_module.predictor import _classify_severity as network_severity
from src.url_module.predictor import _classify_severity as url_severity


SEVERITY_CASES = [
    (0.399, "LOW"),
    (0.40, "MEDIUM"),
    (0.699, "MEDIUM"),
    (0.70, "HIGH"),
]


@pytest.mark.parametrize("probability, expected_severity", SEVERITY_CASES)
@pytest.mark.parametrize(
    "classify_severity",
    [email_severity, url_severity, network_severity],
)
def test_shared_severity_boundaries(
    classify_severity,
    probability,
    expected_severity,
):
    assert classify_severity(probability) == expected_severity
