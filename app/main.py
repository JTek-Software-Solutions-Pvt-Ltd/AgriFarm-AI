"""FastAPI entry point for Kisan Dost."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.agents.graph import KisanDostGraph
from app.api.routes import build_routes
from app.api.voice_routes import router as voice_router
from app.config import get_settings
from app.database import ConversationStore
from app.speech.adapter import SpeechAdapter

settings = get_settings()
logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create shared application services and register chat routes."""
    app.state.settings = settings
    app.state.graph = KisanDostGraph(settings)
    app.state.store = ConversationStore(settings)
    app.state.speech = SpeechAdapter()
    app.include_router(build_routes(app.state.graph, app.state.store, app.state.speech))
    try:
        yield
    finally:
        # Close optional services only when they expose an async/sync close method.
        close = getattr(app.state.graph, "close", None)
        if callable(close):
            try:
                result = close()
                if hasattr(result, "__await__"):
                    await result
            except Exception:
                logging.getLogger(__name__).exception("Error while closing agent graph")


app = FastAPI(title=settings.app_name, version="0.2.2", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8081",
        "http://127.0.0.1:8081",
        "http://localhost:19006",
        "http://127.0.0.1:19006",
        "http://localhost:19000",
        "http://127.0.0.1:19000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(voice_router)


@app.get("/")
async def root() -> dict:
    return {"name": settings.app_name, "status": "running", "docs": "/docs"}


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "environment": settings.app_env,
        "ollama_enabled": settings.ollama_enabled,
        "speech_to_text": "SraVaani (lazy-loaded on first transcription)",
        "asr_model": "ARTPARK-IISc/SraVaani-0.5-live",
        "voice_status_endpoint": "/api/v1/voice/status",
        "voice_transcribe_endpoint": "/api/v1/voice/transcribe",
        "chat_endpoint": "/api/v1/chat",
    }
