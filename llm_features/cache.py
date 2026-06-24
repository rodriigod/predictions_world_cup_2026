"""Caché agresivo en disco para las features de LLM por partido.

Una búsqueda web + llamada al LLM por partido es lenta y cuesta cuota, así que
NO se repite entre corridas: el resultado se guarda como JSON bajo
`files/cache/llm_features/<clave>.json` y se reutiliza para siempre por defecto.

La clave es (team_home, team_away, match_date) normalizada — el mismo cruce en
la misma fecha siempre cae en el mismo archivo. `max_age_days` permite forzar
re-extracción si quieres refrescar (p.ej. lesiones cerca del partido), pero el
default es "sin expiración" (agresivo, como pidió el encargo).
"""

from __future__ import annotations

import hashlib
import json
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from llm_features.schema import MatchFeatures

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "files/cache/llm_features"


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return " ".join(s.lower().split())


def cache_key(team_home: str, team_away: str, match_date: str) -> str:
    raw = f"{_norm(team_home)}|{_norm(team_away)}|{str(match_date).strip()}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _cache_path(team_home: str, team_away: str, match_date: str) -> Path:
    return CACHE_DIR / f"{cache_key(team_home, team_away, match_date)}.json"


def load(team_home: str, team_away: str, match_date: str, *,
         max_age_days: Optional[float] = None) -> Optional[MatchFeatures]:
    """Devuelve las features cacheadas o None si no hay (o expiraron).

    `max_age_days=None` (default) -> nunca expira. Si se da, un archivo más
    viejo que ese umbral se ignora (se forzará re-extracción)."""
    path = _cache_path(team_home, team_away, match_date)
    if not path.exists():
        return None
    if max_age_days is not None:
        age_days = (time.time() - path.stat().st_mtime) / 86400.0
        if age_days > max_age_days:
            return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        feats = MatchFeatures.from_dict(data)
        if feats.source == "llm_web_search":
            feats.source = "cache"      # marca que vino del disco, no de la red
        return feats
    except (json.JSONDecodeError, TypeError, KeyError):
        return None


def save(features: MatchFeatures) -> Path:
    """Persiste las features en disco. Solo cachea extracciones reales
    (`source == 'llm_web_search'`): un esqueleto null no se guarda para no
    fijar un fallo transitorio de red como si fuera un 'no hay datos' real."""
    if features.source != "llm_web_search":
        return _cache_path(features.team_home, features.team_away,
                           features.match_date)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if features.retrieved_at is None:
        features.retrieved_at = datetime.now(timezone.utc).isoformat()
    path = _cache_path(features.team_home, features.team_away,
                       features.match_date)
    path.write_text(json.dumps(features.to_dict(), ensure_ascii=False, indent=2),
                    encoding="utf-8")
    return path
