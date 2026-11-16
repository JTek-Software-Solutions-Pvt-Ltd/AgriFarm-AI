from typing import Literal
from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    language: str = Field(default="en", min_length=2, max_length=20)
    location_name: str | None = Field(default=None, max_length=200)
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)
    session_id: str | None = Field(default=None, max_length=100)

    @field_validator("message")
    @classmethod
    def message_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message must not be blank")
        return value.strip()

    @field_validator("location_name")
    @classmethod
    def location_must_not_be_blank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            return None
        return value.strip() if value else None


class ChatResponse(BaseModel):
    session_id: str
    language: str
    intent: Literal["weather", "crop", "market", "general"]
    response_text: str
    weather: dict | None = None
    model_used: str
    speech_ready: bool = False
    speech_note: str = "Speech model adapter is not configured; use the text response for now."
