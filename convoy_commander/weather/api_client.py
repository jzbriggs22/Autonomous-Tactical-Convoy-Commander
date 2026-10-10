"""Open-Meteo weather API client.

Fetches hourly weather data for a given location. Requires the ``requests``
package (optional dependency).  The caller is expected to handle
``WeatherFetchError`` and fall back to static weather when the API is
unavailable.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

from convoy_commander.weather.compat import require_requests
from convoy_commander.weather.effects import WeatherDataPoint


class WeatherFetchError(Exception):
    """Raised when the weather API call fails."""


_API_URL = "https://api.open-meteo.com/v1/forecast"
_TIMEOUT_S = 5


def fetch_weather_forecast(
    latitude: float,
    longitude: float,
    duration_s: float,
) -> list[WeatherDataPoint]:
    """Fetch hourly weather data from Open-Meteo for the simulation duration.

    Returns a list of WeatherDataPoint sorted by time_offset_s (0 = sim start).
    Raises WeatherFetchError on any failure.
    """
    requests = require_requests()

    now = datetime.now(timezone.utc)
    end = now + timedelta(seconds=duration_s)

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "temperature_2m,precipitation,windspeed_10m,winddirection_10m,visibility",
        "start_date": now.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"),
        "timezone": "UTC",
    }

    try:
        resp = requests.get(_API_URL, params=params, timeout=_TIMEOUT_S)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        raise WeatherFetchError(f"Open-Meteo API request failed: {exc}") from exc

    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    temps = hourly.get("temperature_2m", [])
    precips = hourly.get("precipitation", [])
    winds = hourly.get("windspeed_10m", [])
    wind_dirs = hourly.get("winddirection_10m", [])
    visibilities = hourly.get("visibility", [])

    if not times:
        raise WeatherFetchError("Open-Meteo returned empty hourly data")

    points: list[WeatherDataPoint] = []
    for i, time_str in enumerate(times):
        offset_s = i * 3600.0  # hourly data: each entry is 1 hour apart
        if offset_s > duration_s + 3600:
            break
        points.append(WeatherDataPoint(
            time_offset_s=offset_s,
            temperature_c=float(temps[i]) if i < len(temps) and temps[i] is not None else 20.0,
            precipitation_mm_h=max(0.0, float(precips[i])) if i < len(precips) and precips[i] is not None else 0.0,
            wind_speed_ms=_kmh_to_ms(float(winds[i])) if i < len(winds) and winds[i] is not None else 0.0,
            wind_direction_rad=math.radians(float(wind_dirs[i])) if i < len(wind_dirs) and wind_dirs[i] is not None else 0.0,
            visibility_m=float(visibilities[i]) if i < len(visibilities) and visibilities[i] is not None else 10000.0,
        ))

    if not points:
        raise WeatherFetchError("No valid weather data points parsed")

    return points


def _kmh_to_ms(kmh: float) -> float:
    """Convert km/h to m/s."""
    return kmh / 3.6
