"""Open-Meteo weather client used by Kisan Dost.

This module returns measured/current conditions plus hourly and daily forecast
arrays. It deliberately does not infer official warnings or agricultural
recommendations; those are handled by the advisory/alert layer.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import httpx

from app.config import Settings


class WeatherError(RuntimeError):
    """A safe, user-facing weather lookup failure."""


class OpenMeteoWeather:
    """Open-Meteo geocoding and forecast client."""

    def __init__(self, settings: Settings):
        self.settings = settings

    async def _get_json(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(
                timeout=self.settings.http_timeout_seconds,
                follow_redirects=True,
                headers={"User-Agent": "KisanDost/1.0 weather-advisory"},
            ) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                payload = response.json()
        except httpx.TimeoutException as exc:
            raise WeatherError(
                "Weather provider timed out. Please try again shortly."
            ) from exc
        except httpx.HTTPStatusError as exc:
            # Do not expose provider response bodies or internal URLs to farmers.
            raise WeatherError(
                "Weather provider is temporarily unavailable. Please try again."
            ) from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise WeatherError(
                "Weather data is temporarily unavailable. Please try again."
            ) from exc

        if not isinstance(payload, dict):
            raise WeatherError("Weather provider returned an unexpected response.")
        if payload.get("error"):
            reason = payload.get("reason")
            # Provider validation errors are useful during local development,
            # but keep the farmer-facing message generic.
            raise WeatherError(
                "Weather provider could not process the requested location or forecast."
                if not reason
                else "Weather provider could not process the requested forecast."
            )
        return payload

    async def geocode_candidates(self, location_name: str) -> list[dict[str, Any]]:
        """Return multiple candidates so the caller can validate state/district."""
        query = (location_name or "").strip()
        if not query:
            return []

        payload = await self._get_json(
            self.settings.open_meteo_geocoding_url,
            {
                "name": query,
                "count": 10,
                "language": "en",
                "format": "json",
            },
        )
        results = payload.get("results") or []
        if not isinstance(results, list):
            return []

        candidates: list[dict[str, Any]] = []
        for place in results:
            if not isinstance(place, dict):
                continue
            try:
                latitude = float(place["latitude"])
                longitude = float(place["longitude"])
                if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
                    continue
                candidates.append(
                    {
                        "name": str(place.get("name") or query),
                        "latitude": latitude,
                        "longitude": longitude,
                        "admin1": place.get("admin1"),
                        "admin2": place.get("admin2"),
                        "admin3": place.get("admin3"),
                        "country": place.get("country"),
                        "country_code": place.get("country_code"),
                        "feature_code": place.get("feature_code"),
                        "population": place.get("population"),
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue
        return candidates

    async def geocode(self, location_name: str) -> dict[str, Any]:
        candidates = await self.geocode_candidates(location_name)
        if not candidates:
            raise WeatherError(
                "Could not find that location. Try a nearby village, town or district."
            )
        return candidates[0]

    async def forecast(
        self,
        latitude: float,
        longitude: float,
        location_label: str = "selected location",
        forecast_days: int = 7,
    ) -> dict[str, Any]:
        """Fetch current, hourly and daily data required by the advisory engine."""
        try:
            latitude = float(latitude)
            longitude = float(longitude)
        except (TypeError, ValueError) as exc:
            raise WeatherError("Valid latitude and longitude are required.") from exc

        if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
            raise WeatherError("Latitude or longitude is outside the valid range.")
        if isinstance(forecast_days, bool) or not isinstance(forecast_days, int):
            raise WeatherError("forecast_days must be a whole number between 1 and 16.")
        if not 1 <= forecast_days <= 16:
            raise WeatherError("forecast_days must be between 1 and 16.")

        # Keep these fields aligned with the consumers in weather_routes.py and
        # the existing LangGraph weather node. Do not remove legacy keys.
        current_fields = (
            "temperature_2m,relative_humidity_2m,apparent_temperature,"
            "precipitation,rain,showers,snowfall,weather_code,cloud_cover,"
            "pressure_msl,wind_speed_10m,wind_direction_10m,wind_gusts_10m"
        )
        hourly_fields = (
            "temperature_2m,relative_humidity_2m,precipitation_probability,"
            "precipitation,rain,showers,snowfall,weather_code,cloud_cover,"
            "wind_speed_10m,wind_direction_10m,wind_gusts_10m,"
            "dew_point_2m,et0_fao_evapotranspiration"
        )
        daily_fields = (
            "weather_code,temperature_2m_max,temperature_2m_min,"
            "precipitation_probability_max,precipitation_sum,rain_sum,"
            "wind_speed_10m_max,wind_gusts_10m_max,sunrise,sunset,"
            "et0_fao_evapotranspiration"
        )
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": current_fields,
            "hourly": hourly_fields,
            "daily": daily_fields,
            "timezone": "auto",
            "forecast_days": forecast_days,
            "wind_speed_unit": "kmh",
            "temperature_unit": "celsius",
            "precipitation_unit": "mm",
        }

        payload = await self._get_json(
            self.settings.open_meteo_forecast_url, params
        )

        current = payload.get("current")
        hourly = payload.get("hourly")
        daily = payload.get("daily")
        if not all(isinstance(section, dict) for section in (current, hourly, daily)):
            raise WeatherError(
                "Weather provider returned incomplete current, hourly or daily forecast data."
            )

        # A usable hourly forecast must have aligned time and core weather arrays.
        hourly_times = hourly.get("time") or []
        hourly_temps = hourly.get("temperature_2m") or []
        if not isinstance(hourly_times, list) or not isinstance(hourly_temps, list):
            raise WeatherError("Weather provider returned an invalid hourly forecast.")
        if hourly_times and len(hourly_times) != len(hourly_temps):
            raise WeatherError("Weather provider returned mismatched hourly forecast data.")

        daily_times = daily.get("time") or []
        if not isinstance(daily_times, list):
            raise WeatherError("Weather provider returned an invalid daily forecast.")

        timezone_name = payload.get("timezone") or "UTC"
        return {
            "location": location_label,
            "latitude": latitude,
            "longitude": longitude,
            "timezone": timezone_name,
            "timezone_abbreviation": payload.get("timezone_abbreviation"),
            "utc_offset_seconds": payload.get("utc_offset_seconds"),
            "current": current,
            "current_units": payload.get("current_units") or {},
            "hourly": hourly,
            "hourly_units": payload.get("hourly_units") or {},
            "daily": daily,
            "daily_units": payload.get("daily_units") or {},
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
            label = location_name or "farmer-selected location"
            return await self.forecast(latitude, longitude, label, forecast_days)
        if not location_name or not location_name.strip():
            raise WeatherError("Provide a village/district or GPS coordinates.")

        place = await self.geocode(location_name)
        label = ", ".join(
            str(part)
            for part in (
                place.get("name"),
                place.get("admin2"),
                place.get("admin1"),
                place.get("country"),
            )
            if part
        )
        return await self.forecast(
            place["latitude"], place["longitude"], label, forecast_days
        )


def summarize_weather(weather: dict[str, Any], language: str = "en") -> str:
    """Compatibility formatter used by the existing LangGraph weather node.

    This intentionally remains a compact English formatter for backwards
    compatibility. Farmer-facing localized summaries are produced by the
    weather advisory response layer.
    """
    current = weather.get("current") or {}
    daily = weather.get("daily") or {}
    temp = current.get("temperature_2m")
    humidity = current.get("relative_humidity_2m")
    rain = current.get("precipitation")
    wind = current.get("wind_speed_10m")
    times = daily.get("time") or []
    rain_prob = daily.get("precipitation_probability_max") or []

    parts = [f"Weather for {weather.get('location', 'selected location')}."]
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
