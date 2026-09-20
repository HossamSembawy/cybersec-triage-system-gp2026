"""
FastAPI Application Factory
Registers routers, configures CORS, and exposes the app instance.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes.email import router as email_router
from src.api.routes.url import router as url_router
from src.api.routes.network import router as network_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    app = FastAPI(
        title="Explainable AI Cybersecurity Triage System",
        description=(
            "Multi-model threat correlation and decision support. "
            "CM3070 Final Year Project — University of London."
        ),
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(email_router)
    app.include_router(url_router)
    app.include_router(network_router)

    @app.get("/health", tags=["System"])
    async def health():
        return {
            "status": "ok",
            "service": "cybersec-triage-api",
            "version": "0.1.0",
        }

    logger.info("Application created. Registered routers: email, url, network")
    return app


app = create_app()