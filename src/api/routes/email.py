"""
API Route — Email Phishing Detection
POST /api/v1/email/analyze

Accepts raw email text, runs DistilBERT inference,
and returns a structured triage-ready JSON response.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from src.api.dependencies import get_email_predictor

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/email",
    tags=["Email Detection"]
)


# Request schema
class EmailTextRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=50_000,
        description="Raw email text in text_combined format.",
        examples=[
            "urgent verify your account details immediately "
            "click here your account has been suspended"
        ],
    )


# Response schema
class EmailAnalysisResponse(BaseModel):
    prediction:           str
    phishing_probability: float
    severity:             str
    label:                int
    model_version:        str
    inference_latency_ms: float


# Endpoints
@router.post(
    "/analyze",
    response_model=EmailAnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyse an email for phishing indicators",
    description=(
        "Submit raw email text for phishing classification. "
        "Returns prediction, probability, severity rating, "
        "and inference metadata."
    ),
)
async def analyze_email(
    request: EmailTextRequest,
    predictor=Depends(get_email_predictor),
) -> EmailAnalysisResponse:

    logger.info(
        "Email analysis request received (length=%d chars)",
        len(request.text)
    )

    try:
        result = predictor.predict(request.text)

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
        "Prediction: %s (prob=%.4f, severity=%s, latency=%.1fms)",
        result.prediction,
        result.phishing_probability,
        result.severity,
        result.inference_latency_ms,
    )

    return EmailAnalysisResponse(
        prediction=result.prediction,
        phishing_probability=round(result.phishing_probability, 4),
        severity=result.severity,
        label=result.label,
        model_version=result.model_version,
        inference_latency_ms=round(result.inference_latency_ms, 2),
    )


@router.get(
    "/health",
    status_code=status.HTTP_200_OK,
    summary="Email module health check",
)
async def email_health():
    """Returns the health status of the email detection module."""
    return {
        "status": "ok",
        "module": "email_detection",
        "model_version": "distilbert-phishing-v1",
    }
