"""Chat and streaming voice WebSocket routes for Kisan Dost."""

import asyncio
import json
import logging

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.agents.graph import KisanDostGraph
from app.database import ConversationStore
from app.schemas import ChatRequest, ChatResponse
from app.services.streaming_asr import StreamingASRSession

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
MAX_UTTERANCE_SECONDS = 60
MAX_UTTERANCE_BYTES = SAMPLE_RATE * 4 * MAX_UTTERANCE_SECONDS


def _get_session_id(requested: str | None, store: ConversationStore) -> str:
    from uuid import UUID

    if requested:
        try:
            return str(UUID(requested))
        except (ValueError, TypeError, AttributeError):
            pass

    return store.new_session_id()


def build_routes(
    graph: KisanDostGraph,
    store: ConversationStore,
    speech,
) -> APIRouter:
    """Build chat and live voice routes."""

    router = APIRouter()

    @router.post("/api/v1/chat", response_model=ChatResponse)
    async def chat(body: ChatRequest) -> ChatResponse:
        session_id = _get_session_id(body.session_id, store)
        history = await store.get_history(session_id)

        state = await graph.invoke(
            {
                "session_id": session_id,
                "message": body.message,
                "language": body.language,
                "location_name": body.location_name,
                "latitude": body.latitude,
                "longitude": body.longitude,
                "conversation_history": history,
            }
        )

        response_text = state.get(
            "response_text",
            "Sorry, I could not prepare a response. Please try again.",
        )

        await store.save_turn(
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

        asr_session = None
        session_id = None
        language = "en"
        location_name = None
        latitude = None
        longitude = None
        received_bytes = 0
        last_partial = ""

        await websocket.send_json(
            {
                "type": "ready",
                "audio_input": "binary float32 little-endian PCM",
                "sample_rate": SAMPLE_RATE,
                "channels": 1,
                "tts_ready": bool(getattr(speech, "ready", False)),
                "message": (
                    "Send a JSON start event, binary audio frames, "
                    "and a JSON end event."
                ),
            }
        )

        try:
            while True:
                packet = await websocket.receive()

                if packet["type"] == "websocket.disconnect":
                    break

                raw_audio = packet.get("bytes")

                # Binary WebSocket frames contain raw float32 PCM samples.
                if raw_audio is not None:
                    if asr_session is None:
                        await websocket.send_json(
                            {
                                "type": "error",
                                "message": "Send a start event before audio.",
                            }
                        )
                        continue

                    if len(raw_audio) % 4 != 0:
                        await websocket.send_json(
                            {
                                "type": "error",
                                "message": "Audio frame must contain complete float32 samples.",
                            }
                        )
                        continue

                    received_bytes += len(raw_audio)

                    if received_bytes > MAX_UTTERANCE_BYTES:
                        await websocket.send_json(
                            {
                                "type": "error",
                                "message": "Utterance exceeds the 60-second limit. End this turn and start another.",
                            }
                        )
                        asr_session = None
                        continue

                    pcm = np.frombuffer(raw_audio, dtype="<f4")

                    try:
                        partial = await asyncio.to_thread(
                            asr_session.feed,
                            pcm,
                        )

                        if partial and partial != last_partial:
                            last_partial = partial
                            await websocket.send_json(
                                {
                                    "type": "partial_transcript",
                                    "text": partial,
                                }
                            )
                    except Exception:
                        logger.exception("SraVaani streaming inference failed")
                        await websocket.send_json(
                            {
                                "type": "error",
                                "message": "Speech recognition failed for this audio frame.",
                            }
                        )
                        asr_session = None

                    continue

                raw_text = packet.get("text")
                if raw_text is None:
                    continue

                try:
                    payload = json.loads(raw_text)
                except (ValueError, TypeError):
                    await websocket.send_json(
                        {"type": "error", "message": "Invalid JSON control message."}
                    )
                    continue

                event_type = payload.get("type")

                if event_type == "start":
                    if asr_session is not None:
                        await websocket.send_json(
                            {
                                "type": "error",
                                "message": "Finish the current utterance before starting another.",
                            }
                        )
                        continue

                    language = payload.get("language", "en")
                    if not isinstance(language, str) or not language.strip():
                        language = "en"

                    session_id = _get_session_id(
                        payload.get("session_id"),
                        store,
                    )
                    location_name = payload.get("location_name")
                    latitude = payload.get("latitude")
                    longitude = payload.get("longitude")

                    received_bytes = 0
                    last_partial = ""

                    try:
                        asr_session = await asyncio.to_thread(
                            StreamingASRSession
                        )
                    except Exception:
                        logger.exception("Could not create SraVaani stream")
                        await websocket.send_json(
                            {
                                "type": "error",
                                "message": "Could not start speech recognition.",
                            }
                        )
                        asr_session = None
                        continue

                    await websocket.send_json(
                        {
                            "type": "started",
                            "session_id": session_id,
                            "language": language,
                            "sample_rate": SAMPLE_RATE,
                            "format": "float32",
                        }
                    )

                elif event_type == "end":
                    if asr_session is None:
                        await websocket.send_json(
                            {
                                "type": "error",
                                "message": "There is no active audio utterance.",
                            }
                        )
                        continue

                    current_session = asr_session
                    asr_session = None

                    try:
                        transcript = await asyncio.to_thread(
                            current_session.finish
                        )
                    except Exception:
                        logger.exception("SraVaani finalization failed")
                        await websocket.send_json(
                            {
                                "type": "error",
                                "message": "Could not finalize speech recognition.",
                            }
                        )
                        continue

                    if not transcript:
                        await websocket.send_json(
                            {
                                "type": "no_speech",
                                "message": "No speech was recognized. Please try again.",
                            }
                        )
                        continue

                    await websocket.send_json(
                        {
                            "type": "final_transcript",
                            "text": transcript,
                        }
                    )

                    try:
                        history = await store.get_history(session_id)

                        state = await graph.invoke(
                            {
                                "session_id": session_id,
                                "message": transcript,
                                "language": language,
                                "location_name": location_name,
                                "latitude": latitude,
                                "longitude": longitude,
                                "conversation_history": history,
                            }
                        )

                        answer = state.get(
                            "response_text",
                            "Sorry, I could not prepare a response. Please try again.",
                        )

                        await store.save_turn(
                            session_id,
                            language,
                            transcript,
                            answer,
                            state.get("intent", "general"),
                        )

                        audio_result = None
                        try:
                            audio_result = await speech.synthesize(answer, language)
                        except Exception:
                            logger.exception(
                                "IndicF5 TTS failed; returning text response only"
                            )

                        audio_bytes = (
                            audio_result.audio_bytes
                            if audio_result and audio_result.audio_bytes
                            else None
                        )

                        await websocket.send_json(
                            {
                                "type": "agent_response",
                                "session_id": session_id,
                                "language": language,
                                "intent": state.get("intent", "general"),
                                "text": answer,
                                "weather": state.get("weather"),
                                "model_used": state.get("model_used", "unknown"),
                                "audio_available": audio_bytes is not None,
                                "audio_format": "wav" if audio_bytes else None,
                                "sample_rate": 24000 if audio_bytes else None,
                                "audio_transport": "binary" if audio_bytes else None,
                            }
                        )

                        if audio_bytes:
                            await websocket.send_bytes(audio_bytes)
                    except Exception:
                        logger.exception("Voice agent request failed")
                        await websocket.send_json(
                            {
                                "type": "error",
                                "message": "The agent could not process this message.",
                            }
                        )

                elif event_type == "stop":
                    asr_session = None
                    await websocket.send_json({"type": "stopped"})
                    break

                else:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "message": "Unknown event. Use start, end, or stop.",
                        }
                    )

        except WebSocketDisconnect:
            logger.info("Voice WebSocket disconnected")
        except Exception:
            logger.exception("Unexpected voice WebSocket error")
        finally:
            asr_session = None

    return router
