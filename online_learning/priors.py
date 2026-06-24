"""Priors pre-torneo leídos de core/ (SOLO LECTURA).

El online learning NO parte de cero: arranca de los ratings que core/ ya calculó
con toda la historia hasta el día antes del Mundial (cutoff 2026-06-11). Este
módulo expone esos valores y la normalización de nombres, sin tocar nada de
producción.

Nada de core/ importa este paquete: borrar `online_learning/` no afecta al repo.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from core.data.historical import (NAME_MAP, PI_MU, build_historical_dataset)

CUTOFF = "2026-06-11"            # foto pre-torneo (idéntica a ensemble.predict)
_INV_NAME = {v: k for k, v in NAME_MAP.items()}   # inglés -> español (display)


def to_es(en: str) -> str:
    """Nombre en español para display (inverso de NAME_MAP)."""
    return _INV_NAME.get(en, en)


@lru_cache(maxsize=2)
def _snapshots(cutoff: str = CUTOFF) -> dict:
    return build_historical_dataset(cutoff=cutoff)["snapshots"]


@lru_cache(maxsize=1)
def _alias() -> dict:
    """Mapa flexible nombre(es/en, normalizado) -> nombre inglés canónico."""
    snaps = _snapshots()
    out = {}
    for en in snaps:
        out[en.lower()] = en
    for es, en in NAME_MAP.items():
        out[es.lower()] = en
    return out


def canon(team: str) -> str:
    """Nombre inglés canónico (clave de los snapshots) desde español o inglés."""
    a = _alias()
    key = str(team).strip().lower()
    if key in a:
        return a[key]
    raise KeyError(f"equipo desconocido para online_learning: {team!r} "
                   "(usa un nombre del fixture 2026 o del dataset)")


def pretournament_priors(cutoff: str = CUTOFF) -> dict:
    """{nombre_inglés: prior} con los ratings pre-torneo de core/.

    Cada prior: elo, att (pi ataque), dfn (pi defensa), lam_off / lam_def
    (goles esperados a favor / en contra por partido, de las medias rodantes que
    core/ ya usa — `xg_for_last10` / `xg_against_last10`)."""
    snaps = _snapshots(cutoff)
    out = {}
    for en, h in snaps.items():
        lam_off = float(np.mean(h.gf10)) if h.gf10 else 1.25
        lam_def = float(np.mean(h.ga10)) if h.ga10 else 1.25
        out[en] = {"elo": float(h.elo), "att": float(h.att), "dfn": float(h.dfn),
                   "lam_off": lam_off, "lam_def": lam_def}
    return out


def teams_2026() -> list[str]:
    """Las 48 selecciones del Mundial (nombre inglés canónico)."""
    import pandas as pd
    from pathlib import Path
    csv = Path(__file__).resolve().parents[1] / "files/f0_raw/teams_2026.csv"
    return [canon(t) for t in pd.read_csv(csv)["team"]]
