"""Download and cache NASA FIRMS VIIRS active-fire detections.

Two sources:
  * "api"    - FIRMS area API, needs the FIRMS_MAP_KEY environment variable.
  * "public" - FIRMS public 7-day global CSV (no key), filtered to the area box.
The key is never printed or written to disk.
"""
import io
import os
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

API = "https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/{source}/{area}/{days}"
API_DATED = API + "/{date}"
PUBLIC = {
    "VIIRS_NOAA20_NRT": "https://firms.modaps.eosdis.nasa.gov/data/active_fire/noaa-20-viirs-c2/csv/J1_VIIRS_C2_Global_7d.csv",
    "VIIRS_NOAA21_NRT": "https://firms.modaps.eosdis.nasa.gov/data/active_fire/noaa-21-viirs-c2/csv/J2_VIIRS_C2_Global_7d.csv",
    "VIIRS_SNPP_NRT": "https://firms.modaps.eosdis.nasa.gov/data/active_fire/suomi-npp-viirs-c2/csv/SUOMI_VIIRS_C2_Global_7d.csv",
}
RAW = Path("data/raw")


def parse_area(area: str):
    west, south, east, north = (float(v) for v in area.split(","))
    if not (west < east and south < north):
        raise ValueError("area must be west,south,east,north")
    return west, south, east, north


def _get_csv(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=120)
    if r.status_code != 200 or "Invalid" in r.text[:200]:
        # Never echo the URL: for the API it contains the key.
        raise RuntimeError(f"FIRMS request failed: HTTP {r.status_code}: {r.text[:120].strip()}")
    return pd.read_csv(io.StringIO(r.text))


def download(area: str, days: int, source: str, mode: str = "auto") -> tuple[pd.DataFrame, str]:
    """Return raw detections for the area and how they were obtained. Cached in data/raw."""
    RAW.mkdir(parents=True, exist_ok=True)
    key = os.environ.get("FIRMS_MAP_KEY", "").strip()
    if mode == "auto":
        mode = "api" if key else "public"
    tag = f"{mode}_{source}_{area.replace(',', '_')}_{days}d_{date.today():%Y%m%d}.csv"
    cache = RAW / tag
    if cache.exists():
        return pd.read_csv(cache), f"cache ({cache.name})"

    if mode == "api":
        if not key:
            raise SystemExit("Set FIRMS_MAP_KEY or use --mode public.")
        try:
            df = _get_csv(API.format(key=key, source=source, area=area, days=days))
        except RuntimeError:
            # Some keys/sources limit DAY_RANGE: fetch one day at a time and combine.
            parts = []
            for i in range(days):
                day = date.today() - timedelta(days=i)
                parts.append(_get_csv(API_DATED.format(key=key, source=source, area=area,
                                                       days=1, date=day.isoformat())))
            df = pd.concat(parts, ignore_index=True)
    else:
        if source not in PUBLIC:
            raise SystemExit(f"No public file for {source}; choose one of {list(PUBLIC)}")
        df = _get_csv(PUBLIC[source])
        west, south, east, north = parse_area(area)
        df = df[df.longitude.between(west, east) & df.latitude.between(south, north)]
        if len(df):
            last = pd.to_datetime(df["acq_date"]).max()
            df = df[pd.to_datetime(df["acq_date"]) > last - pd.Timedelta(days=days)]
    df.to_csv(cache, index=False)
    return df, f"{mode} download ({source})"


def drop_low_confidence(df: pd.DataFrame) -> pd.DataFrame:
    """VIIRS confidence is 'l'/'n'/'h' in the API and 'low'/'nominal'/'high' in public files."""
    conf = df["confidence"].astype(str).str.strip().str.lower().str[0]
    return df[conf != "l"].copy()
