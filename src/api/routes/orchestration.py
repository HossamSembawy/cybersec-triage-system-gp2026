"""API route for multi-model threat orchestration."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.dependencies import get_orchestration_service
from src.orchestration.schemas import (
    OrchestrationRequest,
    OrchestrationResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/orchestration",
    tags=["Threat Orchestration"],
)


@router.post(
    "/analyze",
    response_model=OrchestrationResponse,
    status_code=status.HTTP_200_OK,
    summary="Combine detection results into a triage report",
    description=(
        "Submit structured results from one or more detection modules. "
        "Returns a unified severity, evidence summary, and analyst "
        "recommendation without sending raw threat artifacts to Claude."
    ),
)
def analyze_incident(
    request: OrchestrationRequest,
    service=Depends(get_orchestration_service),
) -> OrchestrationResponse:
    logger.info(
        "Orchestration request received (email=%s, url=%s, network=%s)",
        request.email is not None,
        request.url is not None,
        request.network is not None,
    )

    try:
        result = service.analyze(request)
    except ValueError as exc:
        logger.warning("Claude returned an invalid triage response: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Orchestration returned an invalid response.",
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected orchestration service error")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Orchestration service is unavailable.",
        ) from exc

    logger.info(
        "Orchestration complete (severity=%s, latency=%.1fms)",
        result.severity,
        result.inference_latency_ms,
    )
    return result


@router.get(
    "/health",
    status_code=status.HTTP_200_OK,
    summary="Orchestration module health check",
)
def orchestration_health():
    return {
        "status": "ok",
        "module": "threat_orchestration",
    }
