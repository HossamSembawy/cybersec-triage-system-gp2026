"""Validated inputs and outputs for multi-model threat orchestration."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Severity = Literal["LOW", "MEDIUM", "HIGH"]
ModuleName = Literal["email", "url", "network"]


class EmailDetectionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prediction: Literal["legitimate", "phishing"]
    phishing_probability: float = Field(ge=0.0, le=1.0)
    severity: Severity
    label: Literal[0, 1]
    model_version: str = Field(min_length=1)
    inference_latency_ms: float = Field(ge=0.0)


class UrlDetectionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prediction: Literal["benign", "defacement", "phishing", "malware"]
    malicious_probability: float = Field(ge=0.0, le=1.0)
    severity: Severity
    label: int = Field(ge=0)
    class_probabilities: dict[str, float]
    model_version: str = Field(min_length=1)
    inference_latency_ms: float = Field(ge=0.0)


class NetworkDetectionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prediction: Literal["normal", "anomaly"]
    anomaly_probability: float = Field(ge=0.0, le=1.0)
    severity: Severity
    is_anomaly: bool
    model_version: str = Field(min_length=1)
    inference_latency_ms: float = Field(ge=0.0)


class OrchestrationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    email: EmailDetectionResult | None = None
    url: UrlDetectionResult | None = None
    network: NetworkDetectionResult | None = None

    @model_validator(mode="after")
    def require_detection_result(self):
        if self.email is None and self.url is None and self.network is None:
            raise ValueError("At least one detection result is required")
        return self


class OrchestrationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    severity: Severity
    evidence_summary: str = Field(min_length=1, max_length=2_000)
    analyst_recommendation: str = Field(min_length=1, max_length=2_000)
    contributing_modules: list[ModuleName] = Field(min_length=1, max_length=3)
    model_version: str = Field(min_length=1)
    inference_latency_ms: float = Field(ge=0.0)
