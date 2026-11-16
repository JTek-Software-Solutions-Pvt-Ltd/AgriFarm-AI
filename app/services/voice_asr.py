
import os
import logging
from pathlib import Path
from threading import Lock

import numpy as np
import soundfile as sf
import torch
from transformers import AutoModel

logger = logging.getLogger(__name__)

MODEL_ID = "ARTPARK-IISc/SraVaani-0.5-live"
SAMPLE_RATE = 16000

_model = None
_model_lock = Lock()


def get_asr_model():
    """Load the ASR model once and reuse it."""
    global _model

    with _model_lock:
        if _model is None:
            logger.info("Loading SraVaani ASR model on CPU")

            _model = AutoModel.from_pretrained(
                MODEL_ID,
                trust_remote_code=True,
            ).to("cpu").eval()

            logger.info("SraVaani model loaded successfully")

    return _model


def transcribe_audio(audio_path: str) -> str:
    """
    Transcribe an audio file using SraVaani.

    Expected input: readable audio file, preferably 16 kHz mono WAV.
    Returns: transcript text or raises ValueError for invalid/silent audio.
    """
    path = Path(audio_path)

    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    try:
        audio, sample_rate = sf.read(path, dtype="float32")
    except Exception as exc:
        raise ValueError(f"Unable to read audio file: {exc}") from exc

    if audio.size == 0:
        raise ValueError("Audio file is empty.")

    # Convert stereo audio to mono.
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    if sample_rate != SAMPLE_RATE:
        raise ValueError(
            f"Expected {SAMPLE_RATE} Hz audio, received {sample_rate} Hz. "
            "Convert the audio to 16 kHz mono WAV before transcription."
        )

    if not np.isfinite(audio).all():
        raise ValueError("Audio contains invalid numeric values.")

    peak = float(np.max(np.abs(audio)))

    if peak < 1e-5:
        raise ValueError(
            "Audio contains silence. Please record speech and try again."
        )
    duration_seconds = len(audio) / sample_rate

    if duration_seconds < 0.3:
        raise ValueError(
                "Audio is too short. Please record at least 0.3 seconds of speech."
            )
    model = get_asr_model()

    try:
        with torch.inference_mode():
            result = model.transcribe([str(path)])
    except Exception as exc:
        logger.exception("SraVaani transcription failed")
        raise RuntimeError("Speech transcription failed.") from exc

    if not result or not isinstance(result[0], str):
        raise RuntimeError("SraVaani returned an unexpected result.")

    transcript = result[0].strip()

    if not transcript:
        raise ValueError(
            "No speech was recognized. Check the recording and try again."
        )

    return transcript
