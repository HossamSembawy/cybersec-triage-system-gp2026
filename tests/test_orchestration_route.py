from fastapi.testclient import TestClient

from src.api.dependencies import get_orchestration_service
from src.api.main import create_app
from src.orchestration.schemas import OrchestrationResponse


EMAIL_RESULT = {
    "prediction": "phishing",
    "phishing_probability": 0.98,
    "severity": "HIGH",
    "label": 1,
    "model_version": "distilbert-phishing-v1",
    "inference_latency_ms": 32.5,
}


class StubOrchestrationService:
    def __init__(self, error=None):
        self.error = error
        self.request = None

    def analyze(self, request):
        self.request = request
        if self.error is not None:
            raise self.error

        return OrchestrationResponse(
            severity="HIGH",
            evidence_summary=(
                "The email module identified a high-confidence threat."
            ),
            analyst_recommendation=(
                "Quarantine the email and review the sender."
            ),
            contributing_modules=["email"],
            model_version="claude-sonnet-5",
            inference_latency_ms=420.0,
        )


def build_test_client(service):
    app = create_app()
    app.dependency_overrides[get_orchestration_service] = lambda: service
    return TestClient(app)


def test_orchestration_route_returns_triage_report():
    service = StubOrchestrationService()
    client = build_test_client(service)

    response = client.post(
        "/api/v1/orchestration/analyze",
        json={"email": EMAIL_RESULT},
    )

    assert response.status_code == 200
    assert response.json()["severity"] == "HIGH"
    assert response.json()["contributing_modules"] == ["email"]
    assert service.request.email.prediction == "phishing"


def test_orchestration_route_rejects_empty_request():
    client = build_test_client(StubOrchestrationService())

    response = client.post(
        "/api/v1/orchestration/analyze",
        json={},
    )

    assert response.status_code == 422


def test_orchestration_route_handles_invalid_claude_response():
    service = StubOrchestrationService(
        error=ValueError("Invalid structured output")
    )
    client = build_test_client(service)

    response = client.post(
        "/api/v1/orchestration/analyze",
        json={"email": EMAIL_RESULT},
    )

    assert response.status_code == 502
    assert response.json()["detail"] == (
        "Orchestration returned an invalid response."
    )


def test_orchestration_route_handles_service_failure():
    service = StubOrchestrationService(
        error=RuntimeError("API unavailable")
    )
    client = build_test_client(service)

    response = client.post(
        "/api/v1/orchestration/analyze",
        json={"email": EMAIL_RESULT},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "Orchestration service is unavailable."
    )


def test_orchestration_route_reports_missing_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    client = TestClient(create_app())

    response = client.post(
        "/api/v1/orchestration/analyze",
        json={"email": EMAIL_RESULT},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "Orchestration API key is not configured."
    )
