import httpx
from app.config import Settings


class OllamaClient:
    def __init__(self, settings: Settings):
        self.settings = settings

    async def generate(self, prompt: str, system: str) -> str | None:
        if not self.settings.ollama_enabled:
            return None
        url = self.settings.ollama_base_url.rstrip("/") + "/api/chat"
        payload = {
            "model": self.settings.ollama_model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "options": {"temperature": 0.2},
        }
        try:
            async with httpx.AsyncClient(timeout=self.settings.ollama_timeout_seconds) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
            content = data.get("message", {}).get("content")
            return content.strip() if isinstance(content, str) and content.strip() else None
        except (httpx.HTTPError, ValueError, TypeError):
            return None
