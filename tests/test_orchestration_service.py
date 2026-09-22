from types import SimpleNamespace

import pytest

from src.orchestration.schemas import OrchestrationRequest, TriageDecision
from src.orchestration.service import OrchestrationService


class StubMessages:
    def __init__(self, decision):
        self.decision = decision
        self.parse_arguments = None

    def parse(self, **arguments):
        self.parse_arguments = arguments
        return SimpleNamespace(parsed_output=self.decision)


class StubClaudeClient:
    def __init__(self, decision):
        self.messages = StubMessages(decision)


def build_email_request(severity="HIGH"):
    return OrchestrationRequest(
        email={
            "prediction": "phishing",
            "phishing_probability": 0.98,
            "severity": severity,
            "label": 1,
            "model_version": "distilbert-phishing-v1",
            "inference_latency_ms": 32.5,
        }
    )


def build_decision(severity="HIGH", modules=None):
    return TriageDecision(
        severity=severity,
        evidence_summary="The email module identified a high-confidence threat.",
        analyst_recommendation="Quarantine the email and review the sender.",
        contributing_modules=modules or ["email"],
    )


def test_service_returns_validated_triage_response():
    client = StubClaudeClient(build_decision())
    service = OrchestrationService(client)

    response = service.analyze(build_email_request())

    assert response.severity == "HIGH"
    assert response.contributing_modules == ["email"]
    assert response.model_version == "claude-sonnet-5"
    assert response.inference_latency_ms >= 0.0


def test_service_sends_only_structured_detection_results():
    client = StubClaudeClient(build_decision())
    service = OrchestrationService(client)

    service.analyze(build_email_request())

    user_message = client.messages.parse_arguments["messages"][0]["content"]
    assert "phishing_probability" in user_message
    assert "distilbert-phishing-v1" in user_message
    assert "raw_email_text" not in user_message
    assert client.messages.parse_arguments["output_format"] is TriageDecision


def test_service_instructs_claude_not_to_invent_cross_module_links():
    client = StubClaudeClient(build_decision())
    service = OrchestrationService(client)

    service.analyze(build_email_request())

    system_prompt = client.messages.parse_arguments["system"]
    assert "independent observations" in system_prompt
    assert "attack chain" in system_prompt
    assert "normalized anomaly score" in system_prompt


def test_service_rejects_module_that_was_not_supplied():
    decision = build_decision(modules=["email", "network"])
    service = OrchestrationService(StubClaudeClient(decision))

    with pytest.raises(ValueError, match="module that was not supplied"):
        service.analyze(build_email_request())


def test_service_rejects_lower_severity_than_detector():
    decision = build_decision(severity="MEDIUM")
    service = OrchestrationService(StubClaudeClient(decision))

    with pytest.raises(ValueError, match="below the highest detector severity"):
        service.analyze(build_email_request(severity="HIGH"))


def test_service_rejects_missing_structured_output():
    service = OrchestrationService(StubClaudeClient(None))

    with pytest.raises(ValueError, match="no structured triage decision"):
        service.analyze(build_email_request())
