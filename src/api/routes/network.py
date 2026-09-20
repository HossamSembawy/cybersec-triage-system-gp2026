"""
API Route — Network Anomaly Detection
POST /api/v1/network/analyze

Accepts a network flow's features, runs Isolation Forest anomaly detection,
and returns a structured triage-ready JSON response.
"""

from __future__ import annotations

import logging
from typing import Union

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from src.api.dependencies import get_network_predictor

logger = logging.getLogger(__name__)

NETWORK_FEATURE_COUNT = 78

router = APIRouter(
    prefix="/api/v1/network",
    tags=["Network Detection"]
)


# Request schema
class NetworkRequest(BaseModel):
    features: Union[dict[str, float], list[float]] = Field(
        ...,
        description=(
            "Flow features as a {name: value} object (dataset column names), "
            "or an ordered list of the 78 feature values."
        ),
        examples=[[0.0] * NETWORK_FEATURE_COUNT],
    )


# Response schema
class NetworkAnalysisResponse(BaseModel):
    prediction:           str
    anomaly_probability:  float
    severity:             str
    is_anomaly:           bool
    model_version:        str
    inference_latency_ms: float


# Endpoints
@router.post(
    "/analyze",
    response_model=NetworkAnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyse a network flow for anomalies",
    description=(
        "Submit a network flow's features for anomaly detection. Returns "
        "whether the flow is anomalous, a normalised anomaly probability, "
        "a severity rating, and inference metadata."
    ),
)
async def analyze_network(
    request: NetworkRequest,
    predictor=Depends(get_network_predictor),
) -> NetworkAnalysisResponse:

    logger.info(
        "Network analysis request received (%d features)",
        len(request.features)
    )

    try:
        result = predictor.predict(request.features)

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception("Unexpected error during prediction")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model inference failed. Please try again.",
        ) from exc

    logger.info(
        "Prediction: %s (anomaly=%.4f, severity=%s, latency=%.1fms)",
        result.prediction,
        result.anomaly_probability,
        result.severity,
        result.inference_latency_ms,
    )

    return NetworkAnalysisResponse(
        prediction=result.prediction,
        anomaly_probability=round(result.anomaly_probability, 4),
        severity=result.severity,
        is_anomaly=result.is_anomaly,
        model_version=result.model_version,
        inference_latency_ms=round(result.inference_latency_ms, 2),
    )


@router.get(
    "/health",
    status_code=status.HTTP_200_OK,
    summary="Network module health check",
)
async def network_health():
    """Returns the health status of the network detection module."""
    return {
        "status": "ok",
        "module": "network_detection",
    }
