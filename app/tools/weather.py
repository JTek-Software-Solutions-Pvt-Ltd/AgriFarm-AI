from datetime import datetime, timezone
from typing import Any

import httpx

from app.config import Settings


class WeatherError(RuntimeError):
    """A safe, user-facing Open-Meteo lookup failure."""


class OpenMeteoWeather:
    """Open-Meteo geocoding, current conditions, hourly and daily forecast."""

    def __init__(self, settings: Settings):
        self.settings = settings

    async def _get_json(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(
                timeout=self.settings.http_timeout_seconds
            ) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                payload = response.json()
            if not isinstance(payload, dict):
                raise WeatherError("Open-Meteo returned an unexpected response.")
            return payload
        except (httpx.HTTPError, ValueError) as exc:
            raise WeatherError(
                "Open-Meteo weather data is temporarily unavailable. Please try again."
            ) from exc

    async def geocode_candidates(self, location_name: str) -> list[dict[str, Any]]:
        """Return several matching locations so callers can validate admin areas."""
        payload = await self._get_json(
            self.settings.open_meteo_geocoding_url,
            {
                "name": location_name,
                "count": 10,
                "language": "en",
                "format": "json",
            },
        )
        results = payload.get("results") or []
        candidates: list[dict[str, Any]] = []
        for place in results:
            try:
                candidates.append({
                    "name": str(place["name"]),
                    "latitude": float(place["latitude"]),
                    "longitude": float(place["longitude"]),
                    "admin1": place.get("admin1"),
                    "admin2": place.get("admin2"),
                    "admin3": place.get("admin3"),
                    "country": place.get("country"),
                    "country_code": place.get("country_code"),
                    "feature_code": place.get("feature_code"),
                    "population": place.get("population"),
                })
            except (KeyError, TypeError, ValueError):
                continue
        return candidates

    async def geocode(self, location_name: str) -> dict[str, Any]:
        candidates = await self.geocode_candidates(location_name)
        if not candidates:
            raise WeatherError(
                f"Could not find '{location_name}'. Try a nearby village, town or district."
            )
        return candidates[0]

    async def forecast(
        self,
        latitude: float,
        longitude: float,
        location_label: str = "selected location",
        forecast_days: int = 7,
    ) -> dict[str, Any]:
        if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
            raise WeatherError("Latitude or longitude is outside the valid range.")
        if not 1 <= forecast_days <= 16:
            raise WeatherError("forecast_days must be between 1 and 16.")

        params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": (
                "temperature_2m,relative_humidity_2m,apparent_temperature,"
                "precipitation,rain,showers,snowfall,weather_code,cloud_cover,"
                "pressure_msl,wind_speed_10m,wind_direction_10m,wind_gusts_10m"
            ),
            "hourly": (
                "temperature_2m,relative_humidity_2m,precipitation_probability,"
                "precipitation,rain,showers,weather_code,cloud_cover,"
                "wind_speed_10m,wind_gusts_10m,et0_fao_evapotranspiration"
            ),
            "daily": (
                "weather_code,temperature_2m_max,temperature_2m_min,"
                "precipitation_probability_max,precipitation_sum,rain_sum,"
                "wind_speed_10m_max,wind_gusts_10m_max,sunrise,sunset,"
                "et0_fao_evapotranspiration"
            ),
            "timezone": "auto",
            "forecast_days": forecast_days,
        }
        payload = await self._get_json(
            self.settings.open_meteo_forecast_url, params
        )
        if not all(
            isinstance(payload.get(key), dict)
            for key in ("current", "hourly", "daily")
        ):
            raise WeatherError("Open-Meteo returned incomplete current/hourly/daily data.")

        return {
            "location": location_label,
            "latitude": latitude,
            "longitude": longitude,
            "timezone": payload.get("timezone", "unknown"),
            "timezone_abbreviation": payload.get("timezone_abbreviation"),
            "utc_offset_seconds": payload.get("utc_offset_seconds"),
            "current": payload["current"],
            "current_units": payload.get("current_units", {}),
            "hourly": payload["hourly"],
            "hourly_units": payload.get("hourly_units", {}),
            "daily": payload["daily"],
            "daily_units": payload.get("daily_units", {}),
            "source": "Open-Meteo",
            "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        }

    async def for_request(
        self,
        location_name: str | None,
        latitude: float | None,
        longitude: float | None,
        forecast_days: int = 7,
    ) -> dict[str, Any]:
        if (latitude is None) != (longitude is None):
            raise WeatherError(
                "Provide both latitude and longitude, or provide a place name."
            )
        if latitude is not None and longitude is not None:
            label = location_name or f"coordinates ({latitude:.4f}, {longitude:.4f})"
            return await self.forecast(latitude, longitude, label, forecast_days)
        if not location_name:
            raise WeatherError("Share your village/district or provide GPS coordinates.")
        place = await self.geocode(location_name)
        label = ", ".join(
            str(part)
            for part in (
                place["name"],
                place.get("admin2"),
                place.get("admin1"),
                place.get("country"),
            )
            if part
        )
        return await self.forecast(
            place["latitude"], place["longitude"], label, forecast_days
        )


def summarize_weather(weather: dict, language: str = "en") -> str:
    """Compatibility formatter used by the existing LangGraph weather node."""
    current = weather.get("current", {})
    daily = weather.get("daily", {})
    temp = current.get("temperature_2m")
    humidity = current.get("relative_humidity_2m")
    rain = current.get("precipitation")
    wind = current.get("wind_speed_10m")
    times = daily.get("time") or []
    rain_prob = daily.get("precipitation_probability_max") or []
    parts = [
        f"Weather for {weather.get('location', 'selected location')} "
        "(source: Open-Meteo)."
    ]
    if temp is not None:
        parts.append(f"Current temperature: {temp} °C.")
    if humidity is not None:
        parts.append(f"Relative humidity: {humidity}%.")
    if rain is not None:
        parts.append(f"Current precipitation: {rain} mm.")
    if wind is not None:
        parts.append(f"Wind speed: {wind} km/h.")
    if rain_prob:
        parts.append(f"Maximum precipitation probability today: {rain_prob[0]}%.")
    if len(rain_prob) > 1:
        parts.append(f"Maximum precipitation probability tomorrow: {rain_prob[1]}%.")
    if times:
        parts.append(f"Forecast dates: {', '.join(map(str, times[:3]))}.")
    parts.append("Forecasts can change; check updated conditions before farm operations.")
    return " ".join(parts)
