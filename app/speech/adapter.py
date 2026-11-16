from dataclasses import dataclass


@dataclass
class SpeechResult:
    text: str | None = None
    audio_bytes: bytes | None = None
    language: str | None = None


class SpeechAdapterNotConfigured(RuntimeError):
    """Raised until a tested model-specific STT/TTS adapter is configured."""


class SpeechAdapter:
    """Interface for the selected Hugging Face STT/TTS checkpoints.

    Model-specific adapters must implement streaming/chunking semantics and be tested
    against pinned model revisions. This placeholder deliberately does not download a
    large or incompatible checkpoint at application startup.
    """

    @property
    def ready(self) -> bool:
        return False

    async def transcribe(self, audio_bytes: bytes, language: str | None = None) -> SpeechResult:
        raise SpeechAdapterNotConfigured("Speech recognition is not configured. Send a text frame to test the agent, or configure a model-specific STT adapter.")

    async def synthesize(self, text: str, language: str) -> SpeechResult:
        raise SpeechAdapterNotConfigured("Speech synthesis is not configured yet.")
