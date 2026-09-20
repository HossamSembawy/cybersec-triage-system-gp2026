"""
API Route — Malicious URL Detection
POST /api/v1/url/analyze

Accepts a raw URL, runs Random Forest inference over lexical features,
and returns a structured triage-ready JSON response.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from src.api.dependencies import get_url_predictor

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/url",
    tags=["URL Detection"]
)


# Request schema
class UrlRequest(BaseModel):
    url: str = Field(
        ...,
        min_length=1,
        max_length=2048,
        description="Raw URL string to classify.",
        examples=[
            "http://paypa1-secure-login.com/verify?id=1"
        ],
    )


# Response schema
class UrlAnalysisResponse(BaseModel):
    prediction:            str
    malicious_probability: float
    severity:              str
    label:                 int
    class_probabilities:   dict
    model_version:         str
    inference_latency_ms:  float


# Endpoints
@router.post(
    "/analyze",
    response_model=UrlAnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyse a URL for malicious indicators",
    description=(
        "Submit a raw URL for classification into benign, phishing, "
        "defacement, or malware. Returns the predicted threat class, "
        "malicious probability, severity rating, and inference metadata."
    ),
)
async def analyze_url(
    request: UrlRequest,
    predictor=Depends(get_url_predictor),
) -> UrlAnalysisResponse:

    logger.info(
        "URL analysis request received (length=%d chars)",
        len(request.url)
    )

    try:
        result = predictor.predict(request.url)

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
        "Prediction: %s (malicious=%.4f, severity=%s, latency=%.1fms)",
        result.prediction,
        result.malicious_probability,
        result.severity,
        result.inference_latency_ms,
    )

    return UrlAnalysisResponse(
        prediction=result.prediction,
        malicious_probability=round(result.malicious_probability, 4),
        severity=result.severity,
        label=result.label,
        class_probabilities={
            name: round(prob, 4)
            for name, prob in result.class_probabilities.items()
        },
        model_version=result.model_version,
        inference_latency_ms=round(result.inference_latency_ms, 2),
    )


@router.get(
    "/health",
    status_code=status.HTTP_200_OK,
    summary="URL module health check",
)
async def url_health():
    """Returns the health status of the URL detection module."""
    return {
        "status": "ok",
        "module": "url_detection",
    }
