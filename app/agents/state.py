from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    message: str
    language: str
    location_name: str | None
    latitude: float | None
    longitude: float | None
    intent: str
    weather: dict[str, Any] | None
    response_text: str
    model_used: str
    session_id: str
    conversation_history: list[dict[str, str]]
