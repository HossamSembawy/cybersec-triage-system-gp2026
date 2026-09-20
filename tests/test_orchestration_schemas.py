import pytest
from pydantic import ValidationError

from src.orchestration.schemas import (
    EmailDetectionResult,
    NetworkDetectionResult,
    OrchestrationRequest,
    OrchestrationResponse,
    UrlDetectionResult,
)


def build_email_result():
    return EmailDetectionResult(
        prediction="phishing",
        phishing_probability=0.98,
        severity="HIGH",
        label=1,
        model_version="distilbert-phishing-v1",
        inference_latency_ms=32.5,
    )


def build_url_result():
    return UrlDetectionResult(
        prediction="phishing",
        malicious_probability=0.87,
        severity="HIGH",
        label=2,
        class_probabilities={
            "benign": 0.13,
            "defacement": 0.02,
            "phishing": 0.80,
            "malware": 0.05,
        },
        model_version="url-randomforest-v1",
        inference_latency_ms=8.2,
    )


def build_network_result():
    return NetworkDetectionResult(
        prediction="anomaly",
        anomaly_probability=0.78,
        severity="HIGH",
        is_anomaly=True,
        model_version="network-isolationforest-v1",
        inference_latency_ms=11.1,
    )


def test_request_accepts_all_detection_results():
    request = OrchestrationRequest(
        email=build_email_result(),
        url=build_url_result(),
        network=build_network_result(),
    )

    assert request.email.prediction == "phishing"
    assert request.url.prediction == "phishing"
    assert request.network.prediction == "anomaly"


def test_request_accepts_one_detection_result():
    request = OrchestrationRequest(email=build_email_result())

    assert request.email is not None
    assert request.url is None
    assert request.network is None


def test_request_rejects_missing_detection_results():
    with pytest.raises(
        ValidationError,
        match="At least one detection result is required",
    ):
        OrchestrationRequest()


def test_request_rejects_raw_artifacts():
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        OrchestrationRequest(
            email=build_email_result(),
            raw_email_text="Click this link immediately",
        )


def test_detection_result_rejects_probability_outside_range():
    with pytest.raises(ValidationError):
        NetworkDetectionResult(
            prediction="anomaly",
            anomaly_probability=1.2,
            severity="HIGH",
            is_anomaly=True,
            model_version="network-isolationforest-v1",
            inference_latency_ms=11.1,
        )


def test_response_requires_valid_contributing_modules():
    response = OrchestrationResponse(
        severity="HIGH",
        evidence_summary=(
            "The email, URL, and network modules each identified a threat."
        ),
        analyst_recommendation=(
            "Isolate the affected host and investigate the related indicators."
        ),
        contributing_modules=["email", "url", "network"],
        model_version="claude-orchestration-v1",
        inference_latency_ms=420.0,
    )

    assert response.severity == "HIGH"
    assert response.contributing_modules == ["email", "url", "network"]
