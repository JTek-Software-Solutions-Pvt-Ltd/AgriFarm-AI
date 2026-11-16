"""Chat and WebSocket routes for the Kisan Dost API."""

import json
import logging
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.agents.graph import KisanDostGraph
from app.database import ConversationStore
from app.schemas import ChatRequest, ChatResponse

logger = logging.getLogger(__name__)


def _get_session_id(requested: str | None, store: ConversationStore) -> str:
    return requested or store.new_session_id()


def build_routes(graph: KisanDostGraph, store: ConversationStore, speech: Any) -> APIRouter:
    """Build the API router using the already-created app services."""
    router = APIRouter()

    @router.post("/api/v1/chat", response_model=ChatResponse)
    async def chat(body: ChatRequest) -> ChatResponse:
        session_id = _get_session_id(body.session_id, store)
        state = await graph.invoke(
            {
                "session_id": session_id,
                "message": body.message,
                "language": body.language,
                "location_name": body.location_name,
                "latitude": body.latitude,
                "longitude": body.longitude,
            }
        )
        response_text = state.get(
            "response_text", "Sorry, I could not prepare a response. Please try again."
        )
        store.save_turn(
            session_id,
            body.language,
            body.message,
            response_text,
            state.get("intent", "general"),
        )
        return ChatResponse(
            session_id=session_id,
            language=body.language,
            intent=state.get("intent", "general"),
            response_text=response_text,
            weather=state.get("weather"),
            model_used=state.get("model_used", "unknown"),
            speech_ready=bool(getattr(speech, "ready", False)),
        )

    @router.websocket("/api/v1/voice/ws")
    async def voice_ws(websocket: WebSocket) -> None:
        await websocket.accept()
        await websocket.send_json(
            {
                "type": "ready",
                "speech_ready": bool(getattr(speech, "ready", False)),
                "message": (
                    "Send JSON text frames with type='text' and a message field. "
                    "For audio transcription, use POST /api/v1/voice/transcribe."
                ),
            }
        )
        try:
            while True:
                packet = await websocket.receive()
                if packet.get("type") == "websocket.disconnect":
                    break

                raw_text = packet.get("text")
                if raw_text is None:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "message": "Send a JSON text frame; use /api/v1/voice/transcribe for audio.",
                        }
                    )
                    continue

                try:
                    payload = json.loads(raw_text)
                    message = payload.get("message")
                    if payload.get("type") != "text" or not isinstance(message, str) or not message.strip():
                        await websocket.send_json(
                            {
                                "type": "error",
                                "message": "Expected JSON with type='text' and a non-empty message.",
                            }
                        )
                        continue

                    request = ChatRequest(
                        message=message.strip(),
                        language=payload.get("language", "en"),
                        location_name=payload.get("location_name"),
                        latitude=payload.get("latitude"),
                        longitude=payload.get("longitude"),
                        session_id=payload.get("session_id"),
                    )
                    session_id = _get_session_id(request.session_id, store)
                    state = await graph.invoke(
                        {
                            "session_id": session_id,
                            "message": request.message,
                            "language": request.language,
                            "location_name": request.location_name,
                            "latitude": request.latitude,
                            "longitude": request.longitude,
                        }
                    )
                    answer = state.get("response_text", "Please try again.")
                    store.save_turn(
                        session_id,
                        request.language,
                        request.message,
                        answer,
                        state.get("intent", "general"),
                    )
                    await websocket.send_json(
                        {
                            "type": "response",
                            "session_id": session_id,
                            "language": request.language,
                            "intent": state.get("intent", "general"),
                            "text": answer,
                            "weather": state.get("weather"),
                            "model_used": state.get("model_used", "unknown"),
                            "audio_available": False,
                        }
                    )
                except (ValueError, TypeError) as exc:
                    await websocket.send_json({"type": "error", "message": f"Invalid request: {exc}"})
                except Exception:
                    logger.exception("Voice WebSocket agent request failed")
                    await websocket.send_json(
                        {"type": "error", "message": "The agent could not process this message. Please try again."}
                    )
        except WebSocketDisconnect:
            logger.info("Voice WebSocket disconnected")

    return router
