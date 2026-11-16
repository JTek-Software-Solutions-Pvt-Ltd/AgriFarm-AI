import pytest
from pydantic import ValidationError
from app.schemas import ChatRequest


def test_chat_request_rejects_blank_message():
    with pytest.raises(ValidationError):
        ChatRequest(message="   ")


def test_chat_request_validates_coordinates():
    with pytest.raises(ValidationError):
        ChatRequest(message="weather", latitude=100, longitude=0)
