"""Validación de nómina para las features del LLM (PASO PREVIO al ensemble).

Problema que resuelve: el LLM (Qwen local leyendo snippets de DuckDuckGo) a
veces extrae nombres que NO pertenecen a la selección — el caso real visto fue
"Carvajal" (jugador de España) filtrándose a Argentina por ruido de los
snippets. Antes de que cualquier nombre entre al ensemble, se cruza contra la
nómina REAL de esa selección; si no aparece, se DESCARTA y se registra en un
log auditable (`results/ensemble/discarded_extractions.csv`).

Fuente de nómina real (prioridad):
  1. Caché de Transfermarkt por equipo (`files/cache/transfermarkt/<team>.json`),
     que es la data de plantel de `microsim/ingest` — si fue scrapeada/rellenada.
  2. Dataset FIFA-24 por país (`files/cache/fifa24_players.csv`): nombres reales
     por selección (~5.7k jugadores). Cubre 36/48 selecciones del Mundial 2026.
  3. Si no hay nómina para el equipo -> NO se puede validar: la entrada se
     CONSERVA pero se marca `unverified` (no se inventa una nómina vacía que
     descartaría todo).
"""

from __future__ import annotations

import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from core.data.historical import NAME_MAP
from microsim.ingest import load_cached_squad

ROOT = Path(__file__).resolve().parents[1]
FIFA_CSV = ROOT / "files/cache/fifa24_players.csv"
DISCARD_LOG = ROOT / "results/ensemble/discarded_extractions.csv"

_ROSTER_CACHE: dict[str, Optional[set]] = {}
_FIFA_BY_COUNTRY: Optional[dict[str, set]] = None


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return " ".join(s.lower().split())


def _tokens(name: str) -> list[str]:
    return [t for t in _norm(name).split() if len(t) >= 3]


_NAME_RE = re.compile(r"^[A-Za-zÀ-ÿ'.\-]+$")


def looks_like_name(name: str) -> bool:
    """Heurística para descartar BASURA que no es un nombre de jugador (titulares
    de noticias, frases). Un nombre real: 1-4 tokens, sin dígitos ni ':', tokens
    alfabéticos. Filtra cosas como 'Lesiones de última hora 2026: bajas...'."""
    s = str(name).strip()
    if not s or len(s) > 40 or ":" in s or any(c.isdigit() for c in s):
        return False
    toks = s.split()
    if not (1 <= len(toks) <= 4):
        return False
    return all(_NAME_RE.match(t) for t in toks)


def _load_fifa_by_country() -> dict[str, set]:
    """{país_EN: set de tokens de nombre} desde el dataset FIFA-24."""
    global _FIFA_BY_COUNTRY
    if _FIFA_BY_COUNTRY is not None:
        return _FIFA_BY_COUNTRY
    out: dict[str, set] = {}
    if FIFA_CSV.exists():
        df = pd.read_csv(FIFA_CSV)
        for r in df.itertuples():
            country = str(getattr(r, "country", "")).strip()
            if not country:
                continue
            out.setdefault(country, set()).update(_tokens(r.player))
    _FIFA_BY_COUNTRY = out
    return out


def real_roster(team: str) -> Optional[set]:
    """Set de tokens de nombre de la nómina real de `team`, o None si no se
    puede determinar (entonces no se valida). Cacheado en memoria."""
    if team in _ROSTER_CACHE:
        return _ROSTER_CACHE[team]
    tokens: set = set()

    # 1) caché de Transfermarkt (nombres reales por jugador)
    squad = load_cached_squad(team)
    if squad is not None:
        for p in squad.players:
            tokens.update(_tokens(p.name))

    # 2) FIFA-24 por país (mapea nombre ES del fixture -> EN del dataset)
    fifa = _load_fifa_by_country()
    en = NAME_MAP.get(team, team)
    for key in (en, team):
        if key in fifa:
            tokens.update(fifa[key])

    result = tokens or None        # None = no validable
    _ROSTER_CACHE[team] = result
    return result


def name_in_roster(name: str, roster: set) -> bool:
    """¿El nombre extraído corresponde a alguien de la nómina PROPIA? Match por
    token (apellido) — LENIENTE a propósito: queremos CONSERVAR jugadores reales
    aunque vengan con nombre parcial ('Carvajal', 'Daniel Muñoz')."""
    toks = _tokens(name)
    if not toks:
        return False
    return any(t in roster for t in toks)


def strong_match(name: str, roster: set) -> bool:
    """Match FUERTE: ≥2 tokens del nombre están en la nómina (nombre+apellido).
    Se usa SOLO para la contaminación cruzada, para evitar falsos positivos por
    apellido suelto ('Núñez' de Uruguay matcheando un Núñez de Paraguay)."""
    toks = _tokens(name)
    return sum(1 for t in toks if t in roster) >= 2


def _cross_contamination(name: str, own_team: str,
                         teams: Optional[list[str]]) -> Optional[str]:
    """Si el nombre COMPLETO (≥2 tokens) pertenece a OTRA selección 2026 y NO a
    la propia, devuelve esa selección — señal fuerte de filtrado. Requiere match
    fuerte para no descartar jugadores reales por colisión de apellido."""
    if not teams:
        return None
    for other in teams:
        if other == own_team:
            continue
        r = real_roster(other)
        if r and strong_match(name, r):
            return other
    return None


def _log_discards(discards: list[dict]) -> None:
    if not discards:
        return
    DISCARD_LOG.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(discards)
    header = not DISCARD_LOG.exists()
    df.to_csv(DISCARD_LOG, mode="a", header=header, index=False)


def validate_features(features: dict, *, teams: Optional[list[str]] = None,
                      mode: str = "conservative", log: bool = True
                      ) -> tuple[dict, list[dict]]:
    """Limpia un dict de `MatchFeatures` descartando nombres que no son del
    equipo. No muta el original. Solo toca listas de nombres (`lesionados_clave`).

    Por qué conservador (default): FIFA-24 (la fuente de nómina) está INCOMPLETO
    —le faltan titulares actuales (Carvajal, Endo, Lamine...)— así que la regla
    estricta 'descartar si no está en la nómina' borraba lesiones REALES. El
    objetivo del log en vivo es acumular señal LLM, no perderla.

    Políticas por nombre:
      - basura que no parece nombre (titular de noticia) -> SIEMPRE se descarta
        (`not_a_name`).
      - en nómina propia (match por apellido) -> se conserva (verificado).
      - `mode='conservative'` (default): se descarta SOLO si el nombre COMPLETO
        (≥2 tokens) está en OTRA selección 2026 (`cross_contamination:<eq>`);
        si no, se CONSERVA aunque no esté en la nómina propia (`unverified`,
        registrado para auditar). Evita falsos positivos por apellido suelto.
      - `mode='strict'`: comportamiento previo — descarta todo lo que no esté en
        la nómina propia (riesgo de borrar reales si FIFA-24 no los tiene).
    `teams`: las 48 selecciones, necesario para el cross-check."""
    import copy
    feats = copy.deepcopy(features)
    discards: list[dict] = []
    ts = datetime.now(timezone.utc).isoformat()
    home, away = feats.get("team_home", "?"), feats.get("team_away", "?")
    date = feats.get("match_date", "?")

    def _rec(team, nm, reason, kept):
        discards.append({"timestamp": ts, "match": f"{home} vs {away}",
                         "date": date, "team": team, "field": "lesionados_clave",
                         "name": nm, "reason": reason, "kept": kept})

    for side, team in (("home", home), ("away", away)):
        sig = feats.get(side) or {}
        names = sig.get("lesionados_clave")
        if not names:
            continue
        roster = real_roster(team)
        kept = []
        for nm in names:
            if not looks_like_name(nm):                  # basura -> fuera
                _rec(team, nm, "not_a_name", False)
                continue
            if roster is not None and name_in_roster(nm, roster):
                kept.append(nm)                          # verificado
                continue
            other = _cross_contamination(nm, team, teams)
            if other:                                    # de otra selección
                _rec(team, nm, f"cross_contamination:{other}", False)
                continue
            if mode == "strict" and roster is not None:
                _rec(team, nm, "not_in_roster", False)
                continue
            kept.append(nm)                              # conservador: se queda
            _rec(team, nm, "unverified", True)
        sig["lesionados_clave"] = kept or None

    if log:
        _log_discards(discards)
    return feats, discards
