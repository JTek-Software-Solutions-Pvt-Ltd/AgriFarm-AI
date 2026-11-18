from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.weather_routes import build_weather_router
from app.config import Settings


def test_weather_routes_validate_location():
    app = FastAPI()
    app.include_router(build_weather_router(Settings(ollama_enabled=False)))
    client = TestClient(app)
    response = client.get("/api/v1/weather/forecast")
    assert response.status_code == 422


def test_weather_advisory_requires_crop_for_existing_crop():
    app = FastAPI()
    app.include_router(build_weather_router(Settings(ollama_enabled=False)))
    client = TestClient(app)
    response = client.post(
        "/api/v1/weather/advisory",
        json={"farmer_type": "existing_crop", "latitude": 17.385, "longitude": 78.486},
    )
    assert response.status_code == 422
