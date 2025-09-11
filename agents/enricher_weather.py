# agents/enricher_weather.py
from __future__ import annotations
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone

OPEN_METEO = "https://api.open-meteo.com/v1/forecast"

def _nearest_hour_index(times: pd.Series, target: pd.Timestamp) -> int | None:
    """Return index of the time value closest to target; None if empty."""
    if times.empty:
        return None
    # Converting to pandas Timestamps for safe subtraction
    diffs = (times - target).abs()
    return int(diffs.argmin())

def _fetch_hourly_weather(lat: float, lon: float, tz: str, date_from: pd.Timestamp, date_to: pd.Timestamp):
    """
    Call Open-Meteo hourly forecast for the date window that includes event time.
    Returns a dict with 'times', 'temps', 'precip_probs' as pandas Series.
    """
    params = {
        "latitude": float(lat),
        "longitude": float(lon),
        "hourly": "temperature_2m,precipitation_probability",
        "timezone": tz,  #  "America/Toronto"
    }
    r = requests.get(OPEN_METEO, params=params, timeout=15)
    r.raise_for_status()
    data = r.json().get("hourly", {})

    times = pd.to_datetime(pd.Series(data.get("time", [])), errors="coerce")
    temps = pd.Series(data.get("temperature_2m", []), dtype="float64")
    pprob = pd.Series(data.get("precipitation_probability", []), dtype="float64")

    # Ensure equal length; if not, trim to min length
    n = min(len(times), len(temps), len(pprob))
    times, temps, pprob = times.iloc[:n], temps.iloc[:n], pprob.iloc[:n]
    return {"times": times, "temps": temps, "pprob": pprob}

def _approx_for_event(lat: float, lon: float, when: pd.Timestamp, tz: str = "America/Toronto"):
    """
    Get approximate temperature (°C) and precipitation probability (%) for an event start time.
    Returns (temp_c, rain_prob) or (np.nan, np.nan) if unavailable.
    """
    try:
        # Open-Meteo returns the next 7–16 days of hourly forecast depending on endpoint.
        # We request and then pick the hour closest to 'when' in the city's local timezone.
        bundle = _fetch_hourly_weather(lat, lon, tz, when.normalize(), when.normalize())
        idx = _nearest_hour_index(bundle["times"], when)
        if idx is None:
            return (np.nan, np.nan)
        temp_c = float(bundle["temps"].iloc[idx]) if idx < len(bundle["temps"]) else np.nan
        rain_p = float(bundle["pprob"].iloc[idx]) if idx < len(bundle["pprob"]) else np.nan
        # Clean up: cap ranges and round 
        if not np.isnan(rain_p):
            rain_p = max(0.0, min(100.0, rain_p))
        return (round(temp_c, 1) if not np.isnan(temp_c) else np.nan,
                round(rain_p, 0) if not np.isnan(rain_p) else np.nan)
    except Exception:
        # For any network/format error
        return (np.nan, np.nan)

def enrich_weather(df: pd.DataFrame, tz: str = "America/Toronto") -> pd.DataFrame:
    """
    Add two columns to the DataFrame:
      - temp_c: approximate temperature in °C at event start time
      - rain_prob: precipitation probability (%) at event start time

    Requirements:
      - df must contain: 'start_datetime', 'lat', 'lon' (lat/lon can be NaN for some rows)
    """
    if df.empty:
        return df

    out = df.copy()
    if "temp_c" not in out.columns:
        out["temp_c"] = np.nan
    if "rain_prob" not in out.columns:
        out["rain_prob"] = np.nan

    # We only attempt for rows with valid coordinates and start time
    mask = (
        out["start_datetime"].notna()
        & out.get("lat", pd.Series([np.nan]*len(out))).notna()
        & out.get("lon", pd.Series([np.nan]*len(out))).notna()
    )


    for idx, row in out[mask].iterrows():
        when = pd.to_datetime(row["start_datetime"]).tz_localize(None).replace(hour=18, minute=0, second=0, microsecond=0)
        lat, lon = float(row["lat"]), float(row["lon"])
        temp_c, rain_p = _approx_for_event(lat, lon, when, tz=tz)
        out.at[idx, "temp_c"] = temp_c
        out.at[idx, "rain_prob"] = rain_p

    return out
