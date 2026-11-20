import asyncio
import io
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
from transformers import AutoModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("kisandost-tts")

MODEL_ID = "ai4bharat/IndicF5"
SAMPLE_RATE = 24000

BASE_DIR = Path(__file__).resolve().parents[2]
REF_AUDIO = BASE_DIR / "tts_samples/indicf5/prompts/PAN_F_HAPPY_00001.wav"

REF_TEXT = (
    "ਭਹੰਪੀ ਵਿੱਚ ਸਮਾਰਕਾਂ ਦੇ ਭਵਨ ਨਿਰਮਾਣ ਕਲਾ ਦੇ ਵੇਰਵੇ ਗੁੰਝਲਦਾਰ "
    "ਅਤੇ ਹੈਰਾਨ ਕਰਨ ਵਾਲੇ ਹਨ, ਜੋ ਮੈਨੂੰ ਖੁਸ਼ ਕਰਦੇ  ਹਨ।"
)


class SynthesisRequest(BaseModel):
    text: str = Field(min_length=1, max_length=1000)
    language: str = "te"


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not REF_AUDIO.is_file():
        raise RuntimeError(f"IndicF5 reference audio not found: {REF_AUDIO}")

    torch.set_num_threads(2)

    logger.info("Loading IndicF5 model...")
    app.state.model = AutoModel.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
    )
    app.state.synthesis_lock = asyncio.Lock()
    logger.info("IndicF5 model loaded successfully.")

    yield

    app.state.model = None
    logger.info("IndicF5 service stopped.")


app = FastAPI(
    title="KisanDost IndicF5 TTS",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": MODEL_ID,
        "model_loaded": getattr(app.state, "model", None) is not None,
        "sample_rate": SAMPLE_RATE,
    }


def generate_audio(model, text: str) -> bytes:
    with torch.inference_mode():
        audio = model(
            text,
            ref_audio_path=str(REF_AUDIO),
            ref_text=REF_TEXT,
        )

    audio = np.asarray(audio)

    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    else:
        audio = audio.astype(np.float32)

    audio = np.squeeze(audio)

    if audio.ndim != 1 or audio.size == 0:
        raise ValueError("IndicF5 returned invalid audio.")

    if not np.isfinite(audio).all():
        raise ValueError("IndicF5 returned non-finite audio values.")

    buffer = io.BytesIO()
    sf.write(buffer, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


@app.post("/synthesize")
async def synthesize(request: SynthesisRequest):
    text = request.text.strip()

    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    if request.language.lower() not in {"te", "telugu"}:
        raise HTTPException(
            status_code=400,
            detail="Initial TTS integration currently supports Telugu only.",
        )

    try:
        async with app.state.synthesis_lock:
            audio_bytes = await asyncio.to_thread(
                generate_audio,
                app.state.model,
                text,
            )

        return Response(
            content=audio_bytes,
            media_type="audio/wav",
            headers={
                "X-Audio-Sample-Rate": str(SAMPLE_RATE),
                "X-Audio-Channels": "1",
            },
        )

    except Exception as exc:
        logger.exception("IndicF5 synthesis failed")
        raise HTTPException(
            status_code=500,
            detail="Speech generation failed. Check TTS service logs.",
        ) from exc
