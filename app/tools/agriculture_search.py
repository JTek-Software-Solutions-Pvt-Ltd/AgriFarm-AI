"""Agricultural source discovery for Kisan Dost.

Search results are leads, not verified agronomic evidence. This module never
claims that a crop, sowing window, pesticide, fertilizer dose, or treatment is
suitable solely because a search result was returned.

Compatible with the existing weather_routes.py call:
    await AgricultureSearch().research(farmer, crop, current_weather, daily_forecast)
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from typing import Any
from urllib.parse import urlparse

try:
    from ddgs import DDGS
except ImportError:  # Weather advisory must remain usable without optional search.
    DDGS = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# Exact host suffixes only. A random host containing "gov.in" is not trusted.
PRIORITY_HOST_SUFFIXES = (
    "icar.gov.in",
    "agri.telangana.gov.in",
    "pjtsau.edu.in",
    "angrau.ac.in",
    "tnau.ac.in",
    "aau.in",
    "hau.ac.in",
    "pau.edu",
    "uasbangalore.edu.in",
    " kvk",
)

_POSITIVE_CACHE_TTL = 6 * 60 * 60
_NEGATIVE_CACHE_TTL = 5 * 60
_CACHE_MAX_ITEMS = 128
_SEARCH_TIMEOUT_SECONDS = 18
_CACHE: dict[str, tuple[float, list[dict[str, Any]]]] = {}


def _host(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").lower().rstrip(".")
    except ValueError:
        return ""


def _is_priority_source(url: str) -> bool:
    host = _host(url)
    if not host:
        return False
    return any(host == suffix or host.endswith("." + suffix) for suffix in PRIORITY_HOST_SUFFIXES if suffix != " kvk")


def _cache_key(query: str, max_results: int) -> str:
    raw = f"{query.casefold()}|{max_results}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _search_sync(query: str, max_results: int) -> list[dict[str, Any]]:
    if DDGS is None:
        raise RuntimeError("Optional ddgs package is not installed")

    # Keep backend auto: some environments block or throttle individual backends.
    raw = DDGS(timeout=_SEARCH_TIMEOUT_SECONDS).text(
        query,
        region="in-en",
        safesearch="moderate",
        max_results=max_results,
        backend="auto",
    )
    output: list[dict[str, Any]] = []
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        url = str(item.get("href") or "").strip()
        snippet = str(item.get("body") or "").strip()
        if not title or not url or urlparse(url).scheme not in {"http", "https"}:
            continue
        priority = _is_priority_source(url)
        output.append({
            "title": title,
            "url": url,
            "snippet": snippet,
            "is_priority_source": priority,
            "source_trust": "priority_source_domain" if priority else "unverified_source",
            "verification_status": "search_result_not_independently_verified",
        })
    return output


class AgricultureSearch:
    """Cached DDGS discovery helper with short negative-cache expiry."""

    async def search(self, query: str, max_results: int = 5) -> dict[str, Any]:
        key = _cache_key(query, max_results)
        now = time.monotonic()
        cached = _CACHE.get(key)
        if cached:
            ttl = _POSITIVE_CACHE_TTL if cached[1] else _NEGATIVE_CACHE_TTL
            if now - cached[0] < ttl:
                return {"query": query, "cached": True, "results": cached[1]}
            _CACHE.pop(key, None)

        try:
            results = await asyncio.wait_for(
                asyncio.to_thread(_search_sync, query, max_results),
                timeout=_SEARCH_TIMEOUT_SECONDS + 2,
            )
        except Exception as exc:
            # Don't cache transient errors as a long-lived empty result.
            logger.warning("Agriculture search temporarily unavailable: %s", type(exc).__name__)
            return {
                "query": query,
                "cached": False,
                "results": [],
                "search_error": type(exc).__name__,
            }

        if len(_CACHE) >= _CACHE_MAX_ITEMS:
            oldest = min(_CACHE, key=lambda item: _CACHE[item][0])
            _CACHE.pop(oldest, None)
        _CACHE[key] = (time.monotonic(), results)
        return {"query": query, "cached": False, "results": results}

    async def research(
        self,
        farmer: dict[str, Any],
        crop: dict[str, Any] | None,
        current_weather: dict[str, Any],
        daily_forecast: dict[str, Any],
    ) -> dict[str, Any]:
        state = str(farmer.get("state") or "").strip()
        district = str(farmer.get("district") or "").strip()
        village = str(farmer.get("village") or "").strip()
        location = ", ".join(x for x in (district, state, "India") if x)
        month = time.strftime("%B")
        temperature = current_weather.get("temperature_2m")
        humidity = current_weather.get("relative_humidity_2m")
        rainfall = current_weather.get("precipitation")

        if crop and crop.get("name"):
            crop_name = str(crop["name"]).strip()
            scientific_name = str(crop.get("scientific_name") or "").strip()
            identity = " ".join(x for x in (crop_name, scientific_name) if x)
            queries = [
                f'site:icar.gov.in "{crop_name}" package of practices India',
                f'"{identity}" cultivation package of practices {district} {state} KVK agricultural university',
                f'"{crop_name}" irrigation crop stage pest disease management India agricultural university',
            ]
            scenario = "existing_crop"
        else:
            queries = [
                f'crop calendar sowing season {district} {state} India KVK agricultural university',
                f'Rabi Zaid crop planning {district} {state} India state agriculture university',
                f'contingency crop planning rainfall soil {district} {state} India ICAR KVK',
            ]
            scenario = "no_crop"

        # Run searches concurrently, but each request remains independently bounded.
        reports = await asyncio.gather(
            *(self.search(query, max_results=5) for query in queries),
            return_exceptions=True,
        )
        collected: list[dict[str, Any]] = []
        query_reports: list[dict[str, Any]] = []
        failures = 0
        for query, report in zip(queries, reports):
            if isinstance(report, Exception):
                failures += 1
                query_reports.append({"query": query, "cached": False, "result_count": 0})
                continue
            query_reports.append({
                "query": query,
                "cached": bool(report.get("cached")),
                "result_count": len(report.get("results") or []),
            })
            if report.get("search_error"):
                failures += 1
            collected.extend(report.get("results") or [])

        # Deduplicate by canonical URL while retaining the best source label.
        unique: dict[str, dict[str, Any]] = {}
        for result in collected:
            url = result.get("url")
            if not url:
                continue
            previous = unique.get(url)
            if previous is None or result.get("is_priority_source"):
                unique[url] = result
        results = sorted(
            unique.values(),
            key=lambda item: (not item.get("is_priority_source", False), item.get("title", "").casefold()),
        )

        return {
            "provider": "DuckDuckGo",
            "status": "results_found" if results else ("temporarily_unavailable" if failures else "no_results"),
            "research_status": "discovery_leads_only" if results else ("unavailable" if failures else "no_results"),
            "scenario": scenario,
            "location_context": location or "India",
            "village_context": village or None,
            "season_context": month,
            "weather_context": {
                "temperature_c": temperature,
                "relative_humidity_percent": humidity,
                "current_precipitation_mm": rainfall,
                "forecast_dates": (daily_forecast.get("time") or [])[:7],
                "forecast_max_temperature_c": (daily_forecast.get("temperature_2m_max") or [])[:7],
                "forecast_min_temperature_c": (daily_forecast.get("temperature_2m_min") or [])[:7],
                "forecast_rainfall_mm": (daily_forecast.get("precipitation_sum") or [])[:7],
            },
            "queries": query_reports,
            "results": results[:12],
            "important_note": (
                "Search results and snippets are discovery leads, not verified agronomic evidence. "
                "A priority-domain URL does not prove that its page supports a specific crop, "
                "district, season, soil or sowing window. Do not present crop suitability, "
                "pesticide treatments, fertilizer doses or disease diagnoses as verified "
                "without checking the original authoritative document and its relevant content."
            ),
        }
