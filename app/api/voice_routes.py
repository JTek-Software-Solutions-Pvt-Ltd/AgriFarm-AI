
"""Voice transcription endpoints backed by SraVaani ASR."""

import asyncio
import logging
import os
import subprocess
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.services.voice_asr import transcribe_audio

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/voice",
    tags=["Voice"],
)

MAX_AUDIO_BYTES = 20 * 1024 * 1024  # 20 MB

ALLOWED_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".m4a",
    ".webm",
    ".mp4",
    ".ogg",
    ".aac",
    ".3gp",
}

MODEL_ID = "ARTPARK-IISc/SraVaani-0.5-live"


@router.get("/status")
async def voice_status() -> dict:
    """Check voice API configuration without loading the ASR model."""
    return {
        "status": "configured",
        "model": MODEL_ID,
        "endpoint": "/api/v1/voice/transcribe",
        "max_upload_bytes": MAX_AUDIO_BYTES,
    }


@router.post("/transcribe")
async def transcribe(file: UploadFile = File(...)) -> dict:
    """Convert uploaded audio to 16 kHz mono WAV and transcribe it."""

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Audio filename is missing.",
        )

    suffix = Path(file.filename).suffix.lower()

    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=(
                "Unsupported audio format. "
                "Upload WAV, MP3, M4A, WebM, MP4, OGG, AAC, or 3GP."
            ),
        )

    temp_input = None
    temp_wav = None

    try:
        audio_bytes = await file.read(MAX_AUDIO_BYTES + 1)

        if not audio_bytes:
            raise HTTPException(
                status_code=400,
                detail="Uploaded audio is empty.",
            )

        if len(audio_bytes) > MAX_AUDIO_BYTES:
            raise HTTPException(
                status_code=413,
                detail="Audio file exceeds the 20 MB limit.",
            )

        # Save the uploaded file temporarily.
        with tempfile.NamedTemporaryFile(
            suffix=suffix,
            delete=False,
        ) as temp_file:
            temp_file.write(audio_bytes)
            temp_input = temp_file.name

        # SraVaani expects 16 kHz, mono WAV audio.
        temp_wav = str(
            Path(temp_input).with_name(
                Path(temp_input).stem + "_16k_mono.wav"
            )
        )

        try:
            completed = await asyncio.to_thread(
                subprocess.run,
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    temp_input,
                    "-vn",
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    "-c:a",
                    "pcm_s16le",
                    temp_wav,
                ],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )

        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=500,
                detail="FFmpeg is not installed or is missing from PATH.",
            ) from exc

        except subprocess.TimeoutExpired as exc:
            raise HTTPException(
                status_code=408,
                detail="Audio conversion timed out.",
            ) from exc

        if (
            completed.returncode != 0
            or not os.path.isfile(temp_wav)
        ):
            logger.warning(
                "FFmpeg conversion failed: %s",
                (completed.stderr or "")[-1500:],
            )
            raise HTTPException(
                status_code=400,
                detail=(
                    "Could not decode the uploaded audio. "
                    "Please upload a valid recording."
                ),
            )

        # Run transcription without blocking the async event loop.
        transcript = await asyncio.to_thread(
            transcribe_audio,
            temp_wav,
        )

        if not isinstance(transcript, str) or not transcript.strip():
            raise HTTPException(
                status_code=422,
                detail=(
                    "No speech was detected. "
                    "Please record your voice and try again."
                ),
            )

        transcript = transcript.strip()

        # Return both keys for frontend compatibility.
        return {
            "success": True,
            "transcript": transcript,
            "text": transcript,
            "language": "auto",
            "model": MODEL_ID,
        }

    except HTTPException:
        raise

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception("Voice transcription failed")
        raise HTTPException(
            status_code=500,
            detail="Transcription failed. Check the backend logs.",
        ) from exc

    finally:
        await file.close()

        # Remove temporary audio files.
        for path in (temp_input, temp_wav):
            if path and os.path.isfile(path):
                try:
                    os.remove(path)
                except OSError:
                    logger.warning(
                        "Could not remove temporary file: %s",
                        path,
                    )
