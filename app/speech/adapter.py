import os
from dataclasses import dataclass

import httpx


@dataclass
class SpeechResult:
    text: str | None = None
    audio_bytes: bytes | None = None
    language: str | None = None


class SpeechAdapterNotConfigured(RuntimeError):
    """Raised when a speech service is unavailable or not configured."""


class SpeechAdapter:
    def __init__(self):
        self.tts_url = os.getenv(
            "TTS_BASE_URL", str(httpx.URL(scheme="http", host="127.0.0.1", port=8001))
        ).rstrip("/")
        self.timeout = 600.0

    @property
    def ready(self) -> bool:
        return True

    async def transcribe(
        self, audio_bytes: bytes, language: str | None = None
    ) -> SpeechResult:
        raise SpeechAdapterNotConfigured(
            "ASR is handled by the existing SraVaani streaming pipeline."
        )

    async def synthesize(self, text: str, language: str) -> SpeechResult:
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.tts_url}/synthesize",
                    json={"text": text, "language": language},
                )
                response.raise_for_status()

            return SpeechResult(
                text=text,
                audio_bytes=response.content,
                language=language,
            )
        except httpx.HTTPError as exc:
            raise SpeechAdapterNotConfigured(
                f"IndicF5 TTS service request failed: {exc}"
            ) from exc
