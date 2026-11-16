import pytest
from app.config import Settings
from app.tools.weather import OpenMeteoWeather, summarize_weather


@pytest.mark.asyncio
async def test_weather_asks_for_location_when_missing():
    tool = OpenMeteoWeather(Settings())
    with pytest.raises(Exception, match="share your village or district"):
        await tool.for_request(None, None, None)


def test_weather_summary_includes_source_and_forecast():
    weather = {
        "location": "Example, India", "current": {"temperature_2m": 30, "relative_humidity_2m": 50, "precipitation": 0, "wind_speed_10m": 10},
        "daily": {"time": ["2026-10-23", "2026-10-24"], "precipitation_probability_max": [10, 60]},
        "source": "Open-Meteo", "timezone": "Asia/Kolkata"
    }
    result = summarize_weather(weather)
    assert "Open-Meteo" in result
    assert "30 °C" in result
    assert "tomorrow: 60%" in result
