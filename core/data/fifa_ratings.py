"""Calidad por jugador para la microsimulación, desde ratings FIFA-24.

Fuente: github.com/reh1548/FIFA-24-Player-Dataset (player_stats.csv, ~5.7k
jugadores, 135 países). Resuelve el cuello de botella del motor pre-partido:
FBref bloquea el scraping, pero estos ratings (finishing, shot_power, GK…)
SÍ son accesibles y diferencian a los jugadores.

HONESTO: son ratings del videojuego, un PROXY de calidad — no npxG real. Pero
correlacionan con la habilidad y son muchísimo mejor que los defaults por
posición. Sirven para predecir partidos ACTUALES (plantel 2024/25); no para
validar Mundiales viejos (otra época, otros nombres).

Mapeo a las stats que usa src/simulation/match_engine.squad_lambdas:
  npxg_p90      <- ataque (finishing/shot_power/att_position) escalado por pos
  psxg_save_pct <- media de los 4 ratings de portero
  pressures_p90 <- intensidad defensiva (tackle/interceptions/aggression)
"""
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

CSV = Path(__file__).resolve().parents[2] / "files/cache/fifa24_players.csv"
URL = ("https://raw.githubusercontent.com/reh1548/FIFA-24-Player-Dataset/"
       "main/player_stats.csv")

# cuánto del ataque se traduce en xG según la posición del jugador
POS_SCALE = {"FW": 1.0, "WF": 0.95, "AM": 0.80, "MF": 0.45,
             "FB": 0.18, "CB": 0.10, "GK": 0.0}
# calibración global: sin esto, sumar 10 jugadores da ~3 goles/equipo (irreal).
# Con 0.55 un equipo promedio aterriza ~1.3-1.4 xG y uno top ~1.8 (realista).
NPXG_CAL = 0.55

_CACHE = {"by_name": None}


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return " ".join(s.lower().split())


def _ensure_csv() -> None:
    if CSV.exists():
        return
    import urllib.request
    CSV.parent.mkdir(parents=True, exist_ok=True)
    print(f"Descargando ratings FIFA-24 -> {CSV} ...")
    urllib.request.urlretrieve(URL, CSV)


def _load() -> dict:
    if _CACHE["by_name"] is not None:
        return _CACHE["by_name"]
    _ensure_csv()
    df = pd.read_csv(CSV)
    by_name: dict = {}
    last_rows: dict = {}
    for r in df.itertuples():
        key = _norm(r.player)
        by_name[key] = r
        last_rows.setdefault(key.split()[-1], []).append(r)
    # fallback por apellido SOLO si es único (evita mapear al Martínez equivocado)
    _CACHE["by_name"] = by_name
    _CACHE["by_last"] = {k: v[0] for k, v in last_rows.items() if len(v) == 1}
    return by_name


def _f(v, default=np.nan):
    try:
        x = float(v)
        return x if not np.isnan(x) else default
    except (TypeError, ValueError):
        return default


def get_player_stats(name: str, position: str) -> dict:
    """Stats de microsim para un jugador (o {} si no se encuentra)."""
    by_name = _load()
    key = _norm(name)
    row = by_name.get(key)
    if row is None:                                   # fallback por apellido
        row = _CACHE["by_last"].get(key.split()[-1]) if key else None
    if row is None:
        return {}
    atk = np.nanmean([_f(row.finishing), _f(row.shot_power), _f(row.att_position)])
    scale = POS_SCALE.get(position.upper(), 0.45)
    npxg = float(np.clip((atk / 99.0) ** 1.5 * scale * NPXG_CAL, 0.02, 1.0)) if not np.isnan(atk) else 0.0
    gk = np.nanmean([_f(row.gk_reflexes), _f(row.gk_diving),
                     _f(row.gk_handling), _f(row.gk_positioning)])
    save = float(np.clip(0.60 + 0.20 * (gk / 99.0), 0.58, 0.82)) if not np.isnan(gk) else 0.72
    dfn = np.nanmean([_f(row.stand_tackle), _f(row.interceptions), _f(row.aggression)])
    press = float(np.clip(4 + 8 * (dfn / 99.0), 3, 13)) if not np.isnan(dfn) else 0.0
    out = {"npxg_p90": npxg, "pressures_p90": press}
    if position.upper() == "GK":
        out["psxg_save_pct"] = save
    return out


def get_squad_stats_fifa(lineup: list) -> list:
    """[Player] con stats FIFA. Prioridad: stats manuales del JSON > FIFA >
    defaults por posición (mismo contrato que fbref_scraper.get_squad_stats)."""
    from core.simulation.match_engine import Player
    from core.data.fbref_scraper import STAT_FIELDS
    by_name = _load()
    found = sum(1 for p in lineup if _norm(p["name"]) in by_name
                or _norm(p["name"]).split()[-1] in _CACHE["by_last"])
    print(f"\nRatings FIFA-24: {found}/{len(lineup)} jugadores encontrados "
          "(resto usa defaults por posición).")
    players = []
    for p in lineup:
        manual = {k: p[k] for k in STAT_FIELDS if k in p}
        raw = manual or get_player_stats(p["name"], p["position"])
        players.append(Player(
            name=p["name"], position=p["position"],
            npxg_p90=raw.get("npxg_p90", 0.0),
            progressive_carries_p90=raw.get("progressive_carries_p90", 0.0),
            pressures_p90=raw.get("pressures_p90", 0.0),
            tackles_p90=raw.get("tackles_p90", 0.0),
            aerials_won_pct=raw.get("aerials_won_pct", 0.50),
            psxg_save_pct=raw.get("psxg_save_pct", 0.72)))
    return players
