from datetime import date, timedelta
import httpx
from app.config import Settings


class WeatherError(RuntimeError):
    """A safe, user-facing weather lookup failure."""


class OpenMeteoWeather:
    def __init__(self, settings: Settings):
        self.settings = settings

    async def _get_json(self, url: str, params: dict) -> dict:
        try:
            async with httpx.AsyncClient(timeout=self.settings.http_timeout_seconds) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                payload = response.json()
            if not isinstance(payload, dict):
                raise WeatherError("Weather provider returned an unexpected response.")
            return payload
        except (httpx.HTTPError, ValueError) as exc:
            raise WeatherError("Weather data is temporarily unavailable. Please try again shortly.") from exc

    async def geocode(self, location_name: str) -> dict:
        payload = await self._get_json(
            self.settings.open_meteo_geocoding_url,
            {"name": location_name, "count": 1, "language": "en", "format": "json"},
        )
        results = payload.get("results") or []
        if not results:
            raise WeatherError(f"I could not find '{location_name}'. Please try a nearby town or district.")
        place = results[0]
        try:
            return {
                "name": str(place["name"]),
                "latitude": float(place["latitude"]),
                "longitude": float(place["longitude"]),
                "admin1": place.get("admin1"),
                "country": place.get("country"),
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise WeatherError("The location service returned incomplete coordinates.") from exc

    async def forecast(self, latitude: float, longitude: float, location_label: str = "selected location") -> dict:
        payload = await self._get_json(
            self.settings.open_meteo_forecast_url,
            {
                "latitude": latitude,
                "longitude": longitude,
                "current": "temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m",
                "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max,precipitation_sum",
                "timezone": "auto",
                "forecast_days": 3,
            },
        )
        current = payload.get("current")
        daily = payload.get("daily")
        if not isinstance(current, dict) or not isinstance(daily, dict):
            raise WeatherError("Weather provider returned incomplete forecast data.")
        return {
            "location": location_label,
            "latitude": latitude,
            "longitude": longitude,
            "timezone": payload.get("timezone", "unknown"),
            "current": current,
            "daily": daily,
            "source": "Open-Meteo",
            "retrieved_date_utc": date.today().isoformat(),
        }

    async def for_request(self, location_name: str | None, latitude: float | None, longitude: float | None) -> dict:
        if (latitude is None) != (longitude is None):
            raise WeatherError("Please provide both latitude and longitude, or use a place name.")
        if latitude is not None and longitude is not None:
            label = location_name or f"coordinates ({latitude:.3f}, {longitude:.3f})"
            return await self.forecast(latitude, longitude, label)
        if not location_name:
            raise WeatherError("Please share your village or district, or allow location access in the app.")
        place = await self.geocode(location_name)
        label = ", ".join(str(part) for part in [place["name"], place.get("admin1"), place.get("country")] if part)
        return await self.forecast(place["latitude"], place["longitude"], label)


def summarize_weather(weather: dict, language: str = "en") -> str:
    current = weather["current"]
    daily = weather["daily"]
    temp = current.get("temperature_2m")
    rain = current.get("precipitation")
    wind = current.get("wind_speed_10m")
    humidity = current.get("relative_humidity_2m")
    dates = daily.get("time") or []
    probabilities = daily.get("precipitation_probability_max") or []
    today_rain = probabilities[0] if probabilities else None
    tomorrow_rain = probabilities[1] if len(probabilities) > 1 else None
    parts = [f"Weather for {weather['location']} (source: Open-Meteo)."]
    if temp is not None:
        parts.append(f"Current temperature: {temp} °C.")
    if humidity is not None:
        parts.append(f"Relative humidity: {humidity}%.")
    if rain is not None:
        parts.append(f"Current precipitation: {rain} mm.")
    if wind is not None:
        parts.append(f"Wind speed: {wind} km/h.")
    if today_rain is not None:
        parts.append(f"Maximum precipitation probability today: {today_rain}%.")
    if tomorrow_rain is not None:
        parts.append(f"Maximum precipitation probability tomorrow: {tomorrow_rain}%.")
    if dates:
        parts.append(f"Forecast dates: {', '.join(str(d) for d in dates[:3])}.")
    parts.append("Forecasts can change. Check updated conditions before making spraying or irrigation decisions.")
    return " ".join(parts)
