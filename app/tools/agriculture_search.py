"""
DuckDuckGo-backed agricultural research helper for Kisan Dost.

Search results are discovery leads, not verified agronomic recommendations.
The API returns source titles, URLs and snippets so the application can show
provenance and avoid inventing crop suitability or treatment instructions.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from typing import Any

try:
    from ddgs import DDGS
except ImportError:  # Keep weather advisory usable if optional search dependency is absent.
    DDGS = None  # type: ignore[assignment]


TRUSTED_DOMAIN_HINTS = (
    ".icar.gov.in",
    "icar.gov.in",
    ".gov.in",
    "agri.telangana.gov.in",
    "pjtsau.edu.in",
    "angrau.ac.in",
    "tnau.ac.in",
    "aau.in",
    " kvk",
)

_CACHE: dict[str, tuple[float, list[dict[str, Any]]]] = {}
_CACHE_TTL_SECONDS = 6 * 60 * 60
_CACHE_MAX_ITEMS = 128


def _is_trusted_source(url: str) -> bool:
    host = url.lower().split("://")[-1].split("/")[0].split(":")[0]
    return any(hint.strip(". ") in host for hint in TRUSTED_DOMAIN_HINTS)


def _search_sync(query: str, max_results: int) -> list[dict[str, Any]]:
    if DDGS is None:
        raise RuntimeError(
            "DuckDuckGo search dependency is missing. Install it with: pip install ddgs"
        )

    # backend='auto' is intentional: forcing the DuckDuckGo-only backend can
    # return no results even when DDGS automatic search works in the environment.
    raw_results = DDGS(timeout=20).text(
        query,
        region="in-en",
        safesearch="moderate",
        max_results=max_results,
        backend="auto",
    )

    results: list[dict[str, Any]] = []
    for item in raw_results or []:
        title = str(item.get("title") or "").strip()
        url = str(item.get("href") or "").strip()
        snippet = str(item.get("body") or "").strip()
        if not title or not url:
            continue
        results.append({
            "title": title,
            "url": url,
            "snippet": snippet,
            "source_trust": "priority_source_domain" if _is_trusted_source(url) else "unverified_source",
            "verification_status": "search_result_not_independently_verified",
        })
    return results


def _cache_key(query: str) -> str:
    return hashlib.sha256(query.casefold().encode("utf-8")).hexdigest()


class AgricultureSearch:
    """Small, in-process cached agricultural web search provider."""

    async def search(self, query: str, max_results: int = 5) -> dict[str, Any]:
        key = _cache_key(query)
        now = time.monotonic()
        cached = _CACHE.get(key)
        if cached and now - cached[0] < _CACHE_TTL_SECONDS:
            return {
                "query": query,
                "cached": True,
                "results": cached[1],
            }

        results = await asyncio.to_thread(_search_sync, query, max_results)
        if len(_CACHE) >= _CACHE_MAX_ITEMS:
            oldest_key = min(_CACHE, key=lambda item: _CACHE[item][0])
            _CACHE.pop(oldest_key, None)
        _CACHE[key] = (now, results)
        return {
            "query": query,
            "cached": False,
            "results": results,
        }

    async def research(
        self,
        farmer: dict[str, Any],
        crop: dict[str, Any] | None,
        current_weather: dict[str, Any],
        daily_forecast: dict[str, Any],
    ) -> dict[str, Any]:
        state = str(farmer.get("state") or "").strip()
        district = str(farmer.get("district") or "").strip()
        location = ", ".join(part for part in (district, state, "India") if part)
        month = time.strftime("%B")
        temperature = current_weather.get("temperature_2m")
        humidity = current_weather.get("relative_humidity_2m")
        rain = current_weather.get("precipitation")

        if crop and crop.get("name"):
            crop_name = str(crop["name"])
            scientific_name = str(crop.get("scientific_name") or "").strip()
            crop_identity = " ".join(part for part in (crop_name, scientific_name) if part)
            queries = [
                f"{crop_identity} cultivation weather management rainfall heat humidity irrigation India",
                f"{crop_identity} crop protection pest disease management agricultural university India",
                f"{crop_identity} cultivation package of practices {state} India ICAR university KVK",
            ]
            scenario = "existing_crop"
        else:
            weather_context = (
                f"current temperature {temperature} C, relative humidity {humidity} percent, "
                f"current precipitation {rain} mm"
            )
            queries = [
                f"crops to sow {month} {location} seasonal crop calendar agriculture university",
                f"Rabi season suitable crops sowing window {location} India agriculture university KVK",
                f"crop recommendations based on rainfall temperature soil {location} India ICAR",
            ]
            scenario = "no_crop"

        collected: list[dict[str, Any]] = []
        query_reports: list[dict[str, Any]] = []
        for query in queries:
            report = await self.search(query, max_results=5)
            query_reports.append({
                "query": report["query"],
                "cached": report["cached"],
                "result_count": len(report["results"]),
            })
            collected.extend(report["results"])

        # Deduplicate by URL while retaining query order.
        unique: dict[str, dict[str, Any]] = {}
        for result in collected:
            unique.setdefault(result["url"], result)

        results = list(unique.values())
        results.sort(
            key=lambda item: item["source_trust"] != "priority_source_domain"
        )

        return {
            "provider": "DuckDuckGo",
            "status": "results_found" if results else "no_results",
            "scenario": scenario,
            "location_context": location or "India",
            "season_context": month,
            "weather_context": {
                "temperature_c": temperature,
                "relative_humidity_percent": humidity,
                "current_precipitation_mm": rain,
                "forecast_dates": (daily_forecast.get("time") or [])[:7],
                "forecast_max_temperature_c": (daily_forecast.get("temperature_2m_max") or [])[:7],
                "forecast_rainfall_mm": (daily_forecast.get("precipitation_sum") or [])[:7],
            },
            "queries": query_reports,
            "results": results[:12],
            "important_note": (
                "Search snippets are discovery evidence only. They may be outdated, "
                "generic or not specific to the farmer's soil, irrigation access, "
                "variety or crop stage. Verify the linked source and local KVK or "
                "agriculture-department advisory before presenting a crop as suitable "
                "or recommending a pesticide, fertilizer dose or treatment."
            ),
        }
