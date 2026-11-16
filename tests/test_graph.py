import pytest
from app.agents.graph import KisanDostGraph, classify_intent
from app.config import Settings


def test_intent_classifier_routes_common_queries():
    assert classify_intent("Will it rain tomorrow?") == "weather"
    assert classify_intent("What is the mandi price of rice?") == "market"
    assert classify_intent("My paddy leaves have a disease") == "crop"
    assert classify_intent("Namaste") == "general"


@pytest.mark.asyncio
async def test_general_graph_without_external_services(tmp_path):
    settings = Settings(database_path=str(tmp_path / "db.sqlite3"), ollama_enabled=False)
    graph = KisanDostGraph(settings)
    result = await graph.invoke({"message": "Namaste", "language": "en", "location_name": None, "latitude": None, "longitude": None})
    assert result["intent"] == "general"
    assert "Kisan Dost" in result["response_text"]


@pytest.mark.asyncio
async def test_weather_graph_handles_missing_location_without_network(tmp_path):
    settings = Settings(database_path=str(tmp_path / "db.sqlite3"), ollama_enabled=False)
    graph = KisanDostGraph(settings)
    result = await graph.invoke({"message": "Will it rain tomorrow?", "language": "en", "location_name": None, "latitude": None, "longitude": None})
    assert result["intent"] == "weather"
    assert "village or district" in result["response_text"]
    assert result["weather"] is None
