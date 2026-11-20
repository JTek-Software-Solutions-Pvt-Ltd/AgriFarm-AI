"""Native streaming ASR wrapper for SraVaani 0.5 Live."""

import logging

import numpy as np

from app.services.voice_asr import get_asr_model, SAMPLE_RATE

logger = logging.getLogger(__name__)


class StreamingASRSession:
    """Maintains one SraVaani streaming session per user utterance."""

    def __init__(self):
        self.model = get_asr_model()
        self.state = self.model.new_stream(batch_size=1)
        self.finished = False

    def feed(self, pcm: np.ndarray) -> str:
        """Feed one float32 PCM chunk and return the current transcript."""

        if self.finished:
            raise RuntimeError("This ASR session has already finished.")

        audio = np.asarray(pcm, dtype=np.float32).reshape(-1)

        if audio.size == 0:
            return self.model._text(self.state, 0)

        if not np.isfinite(audio).all():
            raise ValueError("Audio contains invalid numeric values.")

        self.model.stream_audio(
            self.state,
            audio,
            sampling_rate=SAMPLE_RATE,
        )

        return self.model._text(self.state, 0).strip()

    def finish(self) -> str:
        """Flush the final audio chunk and return the complete transcript."""

        if self.finished:
            return self.model._text(self.state, 0).strip()

        transcript = self.model.finalize(self.state)
        self.finished = True

        return transcript.strip()
