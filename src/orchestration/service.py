"""Claude-based synthesis of structured detection results."""

from __future__ import annotations

import json
import time

from src.orchestration.schemas import (
    OrchestrationRequest,
    OrchestrationResponse,
    TriageDecision,
)

MODEL_VERSION = "claude-sonnet-5"
MAX_OUTPUT_TOKENS = 1_024

SEVERITY_RANK = {
    "LOW": 0,
    "MEDIUM": 1,
    "HIGH": 2,
}

SYSTEM_PROMPT = """
You are the explanation and synthesis layer in a cybersecurity triage system.
Use only the structured detection results supplied by the specialist models.
Do not perform threat detection and do not invent evidence, indicators, or facts.
Treat email, URL, and network results as independent observations unless explicit
shared identifiers or timestamps are supplied. Do not say that a URL was in an
email, that traffic followed a click, or that the results form an attack chain.
The network anomaly_probability field is a normalized anomaly score, not a
calibrated probability of attack. Describe it as an anomaly score.
Do not infer that a LOW result proves an absence of malicious activity.
The final severity must not be lower than the highest detector severity.
Name only supplied modules in contributing_modules.
Explain the combined evidence clearly and give a specific analyst recommendation.
""".strip()


class OrchestrationService:
    def __init__(self, client, model_version: str = MODEL_VERSION):
        self.client = client
        self.model_version = model_version

    def analyze(
        self,
        request: OrchestrationRequest,
    ) -> OrchestrationResponse:
        start = time.perf_counter()
        detection_results = request.model_dump(exclude_none=True)

        message = self.client.messages.parse(
            model=self.model_version,
            max_tokens=MAX_OUTPUT_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": json.dumps(detection_results, indent=2),
                }
            ],
            output_format=TriageDecision,
        )

        decision = message.parsed_output
        if decision is None:
            raise ValueError("Claude returned no structured triage decision")

        self._validate_decision(decision, detection_results)
        latency_ms = (time.perf_counter() - start) * 1_000

        return OrchestrationResponse(
            severity=decision.severity,
            evidence_summary=decision.evidence_summary,
            analyst_recommendation=decision.analyst_recommendation,
            contributing_modules=decision.contributing_modules,
            model_version=self.model_version,
            inference_latency_ms=latency_ms,
        )

    @staticmethod
    def _validate_decision(
        decision: TriageDecision,
        detection_results: dict,
    ) -> None:
        available_modules = set(detection_results)
        contributing_modules = set(decision.contributing_modules)

        if not contributing_modules.issubset(available_modules):
            raise ValueError("Claude cited a detection module that was not supplied")

        highest_detector_severity = max(
            (result["severity"] for result in detection_results.values()),
            key=SEVERITY_RANK.get,
        )
        highest_detector_rank = SEVERITY_RANK[highest_detector_severity]
        decision_rank = SEVERITY_RANK[decision.severity]

        if decision_rank < highest_detector_rank:
            raise ValueError(
                "Claude returned a severity below the highest detector severity"
            )
