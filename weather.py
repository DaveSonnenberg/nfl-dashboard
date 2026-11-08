"""Game wind for the projection model.

Played games: nflverse schedule wind (mph at kickoff) when the roof was outdoors/open.
Upcoming games: Open-Meteo hourly forecast (free, no key) at the stadium for the kickoff hour.
Domes, closed roofs and retractable roofs (status unknown before the game) count as no wind.
Model input: wind_x = max(wind - 10 mph, 0); unknown -> 0 (neutral).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import requests

CALM = 10.0  # mph; wind below this has no measurable effect
CAP = 8.0    # effect stops growing past CALM + CAP mph (keeps a rare 25-mph game from dominating the fit)

# stadium_id -> (lat, lon, kind)   kind: outdoor | dome | retract
STADIUMS = {
    "ATL97": (33.755, -84.401, "retract"), "BAL00": (39.278, -76.623, "outdoor"), "BOS00": (42.091, -71.264, "outdoor"),
    "BUF00": (42.774, -78.787, "outdoor"), "CAR00": (35.226, -80.853, "outdoor"), "CHI98": (41.862, -87.617, "outdoor"),
    "CIN00": (39.095, -84.516, "outdoor"), "CLE00": (41.506, -81.700, "outdoor"), "DAL00": (32.748, -97.093, "retract"),
    "DEN00": (39.744, -105.020, "outdoor"), "DET00": (42.340, -83.046, "dome"), "GNB00": (44.501, -88.062, "outdoor"),
    "HOU00": (29.685, -95.411, "retract"), "IND00": (39.760, -86.164, "retract"), "JAX00": (30.324, -81.637, "outdoor"),
    "KAN00": (39.049, -94.484, "outdoor"), "LAX01": (33.953, -118.339, "dome"), "MIA00": (25.958, -80.239, "outdoor"),
    "MIN01": (44.974, -93.258, "dome"), "NAS00": (36.166, -86.771, "outdoor"), "NOR00": (29.951, -90.081, "dome"),
    "NYC01": (40.814, -74.074, "outdoor"), "PHI00": (39.901, -75.168, "outdoor"), "PHO00": (33.528, -112.263, "retract"),
    "PIT00": (40.447, -80.016, "outdoor"), "SEA00": (47.595, -122.332, "outdoor"), "SFO01": (37.403, -121.970, "outdoor"),
    "TAM00": (27.976, -82.503, "outdoor"), "VEG00": (36.091, -115.184, "dome"), "WAS00": (38.908, -76.865, "outdoor"),
    # international / neutral sites
    "LON00": (51.556, -0.280, "outdoor"), "LON02": (51.604, -0.066, "outdoor"), "FRA00": (50.069, 8.645, "outdoor"),
    "GER00": (48.219, 11.625, "outdoor"), "MUN01": (48.219, 11.625, "outdoor"), "MEX00": (19.303, -99.150, "outdoor"),
    "SAO00": (-23.545, -46.474, "outdoor"), "RIO00": (-22.912, -43.230, "outdoor"), "MAD01": (40.453, -3.688, "retract"),
    "PAR00": (48.924, 2.360, "outdoor"), "MEL00": (-37.820, 144.983, "outdoor"),
}


def played_wind(sch: pd.DataFrame) -> pd.Series:
    """Wind (mph) per game_id for played games: 0 under a roof, NaN when unknown."""
    roof = sch["roof"].fillna("")
    w = np.where(roof.isin(["dome", "closed"]), 0.0,
                 np.where(roof.isin(["outdoors", "open"]), sch["wind"].astype(float), np.nan))
    return pd.Series(w, index=sch["game_id"].to_numpy())


def forecast_wind(sch: pd.DataFrame, timeout=10) -> dict:
    """Forecast wind (mph) at kickoff for upcoming games at outdoor stadiums. {game_id: mph}.
    One Open-Meteo call per stadium; failures are skipped (that game stays neutral)."""
    up = sch[sch["result"].isna()].copy()
    if up.empty:
        return {}
    kick = pd.to_datetime(up["gameday"] + " " + up["gametime"].fillna("13:00")).dt.tz_localize("America/New_York")
    up["kick_utc"] = kick.dt.tz_convert("UTC")
    up = up[(up["kick_utc"] - pd.Timestamp.now(tz="UTC")) < pd.Timedelta(days=15)]
    out = {}
    for sid, g in up.groupby("stadium_id"):
        lat, lon, kind = STADIUMS.get(sid, (None, None, "retract"))
        if kind != "outdoor":
            out.update({gid: 0.0 for gid in g["game_id"]})
            continue
        try:
            r = requests.get("https://api.open-meteo.com/v1/forecast", timeout=timeout, params={
                "latitude": lat, "longitude": lon, "hourly": "wind_speed_10m", "wind_speed_unit": "mph",
                "timezone": "UTC", "forecast_days": 16})
            r.raise_for_status()
            h = r.json()["hourly"]
            ws = pd.Series(h["wind_speed_10m"], index=pd.to_datetime(h["time"]).tz_localize("UTC"))
        except Exception:
            continue
        for gid, t in zip(g["game_id"], g["kick_utc"]):
            # average over the game window (kickoff to +3h)
            win = ws[(ws.index >= t.floor("h")) & (ws.index <= t + pd.Timedelta(hours=3))]
            if len(win):
                out[gid] = float(win.mean())
    return out


def wind_x(wind) -> np.ndarray:
    w = np.asarray(wind, float)
    return np.where(np.isfinite(w), np.clip(w - CALM, 0.0, CAP), 0.0)
