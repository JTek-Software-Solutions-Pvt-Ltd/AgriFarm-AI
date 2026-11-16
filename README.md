# Kisan Dost AI — runnable starter

A CPU-friendly starter for a multilingual farmer assistant using **Python, FastAPI, LangGraph, Ollama, and Open-Meteo**. It includes a tested weather tool, a LangGraph routing flow, Ollama integration with a deterministic fallback, health/config endpoints, and a WebSocket voice-session protocol scaffold.

> **Honest status:** this is a runnable MVP foundation, not a claim that 22-language streaming STT/TTS is already production-ready. Hugging Face speech checkpoints have different loading/inference APIs and CPU performance. The speech layer is isolated so a model-specific adapter can be added and benchmarked without destabilizing the API/agent. By default, the app starts without downloading large models or requiring Ollama/API credentials.

## Features

- FastAPI REST API and WebSocket endpoint.
- LangGraph state graph with deterministic intent routing.
- Open-Meteo geocoding and forecast retrieval (no API key required for normal non-commercial use; review provider terms before production).
- Optional local Ollama chat model.
- Deterministic fallback responses when Ollama is unavailable.
- SQLite conversation store for basic session history.
- Optional location coordinates supplied by the app; otherwise a village/district name can be used.
- Speech adapter interface with explicit `503` response until a tested STT/TTS adapter is configured. The WebSocket protocol accepts text messages for local end-to-end agent testing and documents the future binary-audio protocol.
- Automated tests that mock external network and model dependencies.

## Requirements

- Python 3.11 or 3.12 recommended.
- 4 GB RAM minimum for the API-only mode; more RAM is recommended when running Ollama and speech models on the same CPU server.
- Optional: Ollama installed locally.

## Windows PowerShell setup

```powershell
cd kisan-dost-ai
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

If `py -3.11` is not installed, use an installed Python 3.11+ version.

## macOS / Linux setup

```bash
cd kisan-dost-ai
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open `http://127.0.0.1:8000/docs` for interactive API docs.

## Optional Ollama setup

Install Ollama, then:

```bash
ollama pull qwen3:1.7b
```

Set these values in `.env`:

```dotenv
OLLAMA_ENABLED=true
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=qwen3:1.7b
```

The app still starts if Ollama is not running; it returns a transparent deterministic fallback instead. The fallback is for development, not a replacement for model-backed agricultural reasoning.

## API quick tests

Health:
```bash
curl http://127.0.0.1:8000/health
```

Chat using a place name:
```bash
curl -X POST http://127.0.0.1:8000/api/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"Will it rain tomorrow?","language":"en","location_name":"Hyderabad, Telangana, India"}'
```

Chat using coordinates (use the farmer's GPS only after the user grants location permission):
```bash
curl -X POST http://127.0.0.1:8000/api/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"Tell me the weather forecast","language":"en","latitude":17.385,"longitude":78.4867}'
```

Run tests:
```bash
pytest -q
```

## WebSocket protocol (MVP)

Connect to `ws://127.0.0.1:8000/api/v1/voice/ws` and send a JSON text frame:

```json
{"type":"text","message":"Will it rain tomorrow?","language":"en","location_name":"Hyderabad, Telangana, India"}
```

The server returns a JSON result. This validates the conversational path without pretending that raw audio has been transcribed. Binary audio frames are currently rejected with a clear message until a tested speech adapter is installed.

## Speech model integration plan

Target shortlist for evaluation:

- STT: AI4Bharat IndicConformer; compare Indic-Transcribe if CPU/RAM allows.
- TTS: AI4Bharat Indic-Speak; compare IndicF5 for voice quality and speed.

Do not enable a model merely by setting an arbitrary Hugging Face repository ID: confirm its license, model card, supported languages, inference API, and CPU requirements first. The next implementation step is to build and benchmark a model-specific adapter against the exact pinned checkpoint. For a 22-language product, keep a language evaluation matrix and test accents/noisy outdoor speech, code-switching, and agricultural vocabulary.

## Safety and production notes

- Weather responses report the source forecast; they do not promise that conditions will be safe for spraying or other high-impact farm activity.
- Add a trusted crop knowledge source before giving crop-disease treatment recommendations.
- Add a verified mandi-price provider and include market, commodity, unit, and timestamp before enabling market advice.
- Add authentication, rate limits, privacy/retention controls, structured logs, timeouts, retries, and concurrency/load tests before public deployment.
- Never infer farm location from IP. Ask permission for device location or ask the farmer to name a place.
