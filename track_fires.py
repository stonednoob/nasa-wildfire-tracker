"""Group NASA FIRMS VIIRS hotspots into individual fires with DBSCAN and track them daily.

Usage:
  python track_fires.py --area -125,32,-114,42 --days 5
  python track_fires.py --mode api --source VIIRS_SNPP_NRT     # needs FIRMS_MAP_KEY
"""
import argparse
import json
from pathlib import Path

import folium
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, QhullError
from sklearn.cluster import DBSCAN

import firms

PINK, INK, GREY = "#E8447A", "#3A1C2B", "#BDBDBD"
EARTH_RADIUS_KM = 6371.0
PALETTE = ["#E8447A", "#3A1C2B", "#F28C28", "#2A9D8F", "#7B2CBF", "#1D4ED8", "#B5179E",
           "#8D6E63", "#C1121F", "#588157", "#0096C7", "#FFB703"]


def cluster(df: pd.DataFrame, eps_km: float, min_samples: int) -> np.ndarray:
    """DBSCAN with great-circle (haversine) distance; inputs and eps both in radians."""
    coords = np.radians(df[["latitude", "longitude"]].to_numpy())
    return DBSCAN(eps=eps_km / EARTH_RADIUS_KM, min_samples=min_samples,
                  metric="haversine", algorithm="ball_tree").fit_predict(coords)


def hull_area_km2(lat, lon) -> float:
    """Convex-hull area after a local equirectangular projection to km."""
    if len(lat) < 3:
        return 0.0
    lat0 = np.radians(np.mean(lat))
    x = EARTH_RADIUS_KM * np.radians(lon) * np.cos(lat0)
    y = EARTH_RADIUS_KM * np.radians(lat)
    try:
        return float(ConvexHull(np.column_stack([x, y])).volume)  # 2-D "volume" is area
    except QhullError:  # all points collinear or identical
        return 0.0


def bearing_deg(lat1, lon1, lat2, lon2) -> float:
    p1, p2, dl = np.radians(lat1), np.radians(lat2), np.radians(lon2 - lon1)
    x = np.sin(dl) * np.cos(p2)
    y = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    return float((np.degrees(np.arctan2(x, y)) + 360) % 360)


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = (np.sin((p2 - p1) / 2) ** 2
         + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2)
    return float(2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a)))


def compass(deg: float) -> str:
    return ["N", "NE", "E", "SE", "S", "SW", "W", "NW"][int((deg + 22.5) // 45) % 8]


def daily_summary(fires: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (fid, day), g in fires.groupby(["fire_id", "acq_date"]):
        rows.append({"fire_id": int(fid), "date": day, "hotspots": len(g),
                     "total_frp_mw": round(float(g["frp"].sum()), 1),
                     "center_lat": round(float(g["latitude"].mean()), 5),
                     "center_lon": round(float(g["longitude"].mean()), 5),
                     "area_km2": round(hull_area_km2(g["latitude"].to_numpy(),
                                                     g["longitude"].to_numpy()), 3)})
    return pd.DataFrame(rows).sort_values(["fire_id", "date"]).reset_index(drop=True)


def fire_overview(fires: pd.DataFrame, summary: pd.DataFrame, n_days_total: int) -> pd.DataFrame:
    rows = []
    for fid, s in summary.groupby("fire_id"):
        g = fires[fires.fire_id == fid]
        first, last = s.iloc[0], s.iloc[-1]
        move = haversine_km(first.center_lat, first.center_lon, last.center_lat, last.center_lon)
        direction = compass(bearing_deg(first.center_lat, first.center_lon,
                                        last.center_lat, last.center_lon)) if move >= 0.5 else "-"
        frp_cv = float(s.total_frp_mw.std(ddof=0) / s.total_frp_mw.mean()) if len(s) > 1 else np.nan
        # Heuristic for non-wildfire heat sources (gas flares, industry): seen on every day,
        # centre barely moves, steady intensity.
        static = (len(s) == n_days_total and n_days_total >= 3 and move < 0.5
                  and g.latitude.std() * 111 < 0.6 and frp_cv < 0.5)
        rows.append({"fire_id": int(fid), "first_day": first.date, "last_day": last.date,
                     "days_active": len(s), "hotspots": len(g),
                     "total_frp_mw": round(float(g.frp.sum()), 1),
                     "max_daily_area_km2": float(s.area_km2.max()),
                     "center_lat": round(float(g.latitude.mean()), 4),
                     "center_lon": round(float(g.longitude.mean()), 4),
                     "spread_km": round(move, 2), "spread_direction": direction,
                     "possible_static_source": bool(static)})
    return pd.DataFrame(rows).sort_values("hotspots", ascending=False).reset_index(drop=True)


def make_map(df: pd.DataFrame, overview: pd.DataFrame, path: Path):
    center = [df.latitude.mean(), df.longitude.mean()]
    m = folium.Map(location=center, zoom_start=6, tiles="OpenStreetMap")
    noise = folium.FeatureGroup(name="Noise (isolated hotspots)")
    for r in df[df.fire_id == -1].itertuples():
        folium.CircleMarker([r.latitude, r.longitude], radius=2, color=GREY, fill=True,
                            fill_opacity=0.7, weight=0).add_to(noise)
    noise.add_to(m)
    fires_layer = folium.FeatureGroup(name="Fires (DBSCAN clusters)")
    for i, fire in enumerate(overview.itertuples()):
        color = PALETTE[i % len(PALETTE)]
        g = df[df.fire_id == fire.fire_id]
        popup = (f"<b>Fire {fire.fire_id}</b><br>Days active: {fire.days_active} "
                 f"({fire.first_day} to {fire.last_day})<br>Hotspots: {fire.hotspots}<br>"
                 f"Total FRP: {fire.total_frp_mw:,.0f} MW<br>Spread: {fire.spread_km} km "
                 f"{fire.spread_direction}"
                 + ("<br><i>Possible static heat source</i>" if fire.possible_static_source else ""))
        for r in g.itertuples():
            folium.CircleMarker([r.latitude, r.longitude], radius=3, color=color, fill=True,
                                fill_opacity=0.85, weight=0,
                                popup=folium.Popup(popup, max_width=260)).add_to(fires_layer)
    fires_layer.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.save(str(path))


def static_map(df: pd.DataFrame, overview: pd.DataFrame, area: str, path: Path, label_top: int = 5):
    """PNG preview of the clusters (the interactive version is fires_map.html)."""
    west, south, east, north = firms.parse_area(area)
    fig, ax = plt.subplots(figsize=(7, 7.5))
    noise = df[df.fire_id == -1]
    ax.scatter(noise.longitude, noise.latitude, s=6, color=GREY, label=f"Noise ({len(noise)})")
    for i, fid in enumerate(overview.fire_id):
        g = df[df.fire_id == fid]
        ax.scatter(g.longitude, g.latitude, s=12, color=PALETTE[i % len(PALETTE)],
                   label=f"{len(overview)} fires (DBSCAN clusters)" if i == 0 else None)
    for fire in overview.head(label_top).itertuples():
        ax.annotate(f"Fire {fire.fire_id}", (fire.center_lon, fire.center_lat), xytext=(6, 6),
                    textcoords="offset points", color=INK, fontsize=9)
    ax.set(xlim=(west, east), ylim=(south, north), xlabel="Longitude", ylabel="Latitude",
           title="FIRMS VIIRS hotspots grouped into fires")
    ax.set_aspect(1 / np.cos(np.radians((south + north) / 2)))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.title.set_color(INK)
    ax.legend(frameon=False, labelcolor=INK, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def top_fires_chart(summary: pd.DataFrame, overview: pd.DataFrame, path: Path, k: int = 5):
    top = overview.head(k)
    dates = sorted(summary.date.unique())
    fig, ax = plt.subplots(figsize=(9, 4.8))
    colors = [PINK, INK, "#F28C28", "#2A9D8F", "#7B2CBF"]
    for color, fid in zip(colors, top.fire_id):
        s = summary[summary.fire_id == fid].set_index("date").reindex(dates, fill_value=0)
        ax.plot(dates, s.hotspots, marker="o", color=color, lw=2, label=f"Fire {fid}")
    ax.set(xlabel="Date (UTC)", ylabel="Hotspots per day", title=f"{len(top)} largest fires")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK)
    ax.tick_params(colors=INK)
    ax.title.set_color(INK)
    ax.legend(frameon=False, labelcolor=INK)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--area", default="-125,32,-114,42", help="west,south,east,north in degrees")
    ap.add_argument("--days", type=int, default=5)
    ap.add_argument("--source", default="VIIRS_NOAA20_NRT",
                    help="VIIRS_NOAA20_NRT (default), VIIRS_SNPP_NRT or VIIRS_NOAA21_NRT")
    ap.add_argument("--mode", choices=["auto", "api", "public"], default="auto",
                    help="auto = API if FIRMS_MAP_KEY is set, else public 7-day file")
    ap.add_argument("--eps-km", type=float, default=1.5)
    ap.add_argument("--min-samples", type=int, default=3)
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()
    firms.parse_area(args.area)
    out = Path(args.out)
    out.mkdir(exist_ok=True)

    raw, how = firms.download(args.area, args.days, args.source, args.mode)
    print(f"Data: {how}; {len(raw)} raw detections")
    if raw.empty:
        raise SystemExit("No detections in this box for these days. Pick a busier region.")
    df = firms.drop_low_confidence(raw).reset_index(drop=True)
    print(f"After dropping low-confidence detections: {len(df)} rows "
          f"({len(raw) - len(df)} removed)")
    days = sorted(df.acq_date.unique())
    print(f"Dates covered: {days[0]} to {days[-1]} ({len(days)} days)")

    # All days clustered together so a growing fire keeps one ID.
    df["fire_id"] = cluster(df, args.eps_km, args.min_samples)
    fires = df[df.fire_id >= 0]
    n_fires = fires.fire_id.nunique()
    n_noise = int((df.fire_id == -1).sum())
    print(f"DBSCAN eps={args.eps_km} km ({args.eps_km / EARTH_RADIUS_KM:.3e} rad), "
          f"min_samples={args.min_samples}: {n_fires} fires, {n_noise} noise hotspots")
    if n_fires == 0:
        raise SystemExit("No clusters found; try a busier region or smaller min_samples.")

    summary = daily_summary(fires)
    overview = fire_overview(fires, summary, len(days))
    summary.to_csv(out / "fires_summary.csv", index=False)
    overview.to_csv(out / "fires_overview.csv", index=False)
    make_map(df, overview, out / "fires_map.html")
    top_fires_chart(summary, overview, out / "top5_fires.png")
    static_map(df, overview, args.area, out / "fires_map.png")

    widest = max(hull_area_km2(g.latitude.to_numpy(), g.longitude.to_numpy())
                 for _, g in fires.groupby("fire_id"))
    print("\nLargest fires:")
    print(overview.head(5)[["fire_id", "days_active", "hotspots", "total_frp_mw",
                            "center_lat", "center_lon", "spread_km", "spread_direction",
                            "possible_static_source"]].to_string(index=False))
    static = overview[overview.possible_static_source]
    print(f"\nPossible static heat sources (not wildfires): {len(static)}"
          + (f" -> fire IDs {static.fire_id.tolist()}" if len(static) else ""))
    print(f"Largest single-cluster hull area: {widest:.1f} km2")

    meta = {"area": args.area, "days_requested": args.days, "source": args.source, "data": how,
            "raw_rows": int(len(raw)), "rows_after_confidence_filter": int(len(df)),
            "dates": [str(d) for d in days], "eps_km": args.eps_km,
            "eps_rad": args.eps_km / EARTH_RADIUS_KM, "min_samples": args.min_samples,
            "n_fires": int(n_fires), "n_noise": n_noise,
            "hotspots_in_fires": int(len(fires)),
            "possible_static_sources": static.fire_id.tolist(),
            "largest_cluster_hull_km2": round(widest, 2)}
    (out / "run_info.json").write_text(json.dumps(meta, indent=2))
    print(f"\nSaved map, CSVs and chart to {out}/")


if __name__ == "__main__":
    main()
