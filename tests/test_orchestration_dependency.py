import sys
from types import SimpleNamespace

import pytest

from src.api import dependencies
from src.orchestration.service import OrchestrationService


class StubAnthropic:
    api_key = None

    def __init__(self, api_key):
        StubAnthropic.api_key = api_key


@pytest.fixture(autouse=True)
def clear_orchestration_service_cache():
    dependencies._load_orchestration_service.cache_clear()
    yield
    dependencies._load_orchestration_service.cache_clear()


def test_dependency_requires_anthropic_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY is required"):
        dependencies.get_orchestration_service()


def test_dependency_creates_cached_orchestration_service(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key")
    monkeypatch.setenv("ANTHROPIC_MODEL", "test-claude-model")
    monkeypatch.setitem(
        sys.modules,
        "anthropic",
        SimpleNamespace(Anthropic=StubAnthropic),
    )

    first_service = dependencies.get_orchestration_service()
    second_service = dependencies.get_orchestration_service()

    assert isinstance(first_service, OrchestrationService)
    assert first_service is second_service
    assert first_service.model_version == "test-claude-model"
    assert StubAnthropic.api_key == "test-api-key"
