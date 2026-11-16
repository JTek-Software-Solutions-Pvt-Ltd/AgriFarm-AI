
import torch
import soundfile as sf
from pathlib import Path
from transformers import AutoModel

MODEL_ID = "ARTPARK-IISc/SraVaani-0.5-live"
AUDIO_FILE = Path(r"D:\kisan-dost-ai\test_speech.wav")

if not AUDIO_FILE.exists():
    raise FileNotFoundError(f"Audio file not found: {AUDIO_FILE}")

audio, sample_rate = sf.read(AUDIO_FILE)

print("Audio file:", AUDIO_FILE)
print("Sample rate:", sample_rate)
print("Shape:", audio.shape)
print("Duration:", len(audio) / sample_rate, "seconds")
print("Peak amplitude:", abs(audio).max())
print("RMS amplitude:", (audio**2).mean() ** 0.5)

print("\nLoading SraVaani on CPU...")
model = AutoModel.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
).to("cpu").eval()

print("\nRunning transcription...")
with torch.inference_mode():
    result = model.transcribe([str(AUDIO_FILE)])

print("\nRaw result:", repr(result))
print("Result type:", type(result))
if result:
    print("First result:", repr(result[0]))
