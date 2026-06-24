"""Ingesta de valores de mercado de plantel para la microsimulación.

OBJETIVO: obtener, por selección, una lista de jugadores con su VALOR DE
MERCADO y su POSICIÓN, que `strength.py` luego agrega en índices de fuerza
ofensiva/defensiva.

HONESTIDAD SOBRE LAS FUENTES (lee esto antes de confiar en el modelo)
--------------------------------------------------------------------
1. **Transfermarkt** es la fuente ideal (valor de mercado por jugador, con
   posición), pero NO es scrapeable de forma confiable y desatendida:
   responde 403 a clientes sin navegador, cambia el HTML, y un scraping
   agresivo es a la vez frágil y poco respetuoso. Aquí implementamos un
   scraper RESPETUOSO (User-Agent real, rate-limiting, caché en disco,
   back-off) que funciona si le das la URL de la plantilla — pero asume que
   muchas peticiones fallarán y degrada con elegancia (devuelve None).
2. **FotMob** es la alternativa: expone ratings por jugador vía una API
   JSON interna. Mismo problema de fragilidad/ToS; se documenta el punto de
   entrada `fetch_fotmob_squad` como stub honesto, no como algo "resuelto".
3. **Fallback que SIEMPRE funciona (el que usan los tests y producción por
   defecto):**
     a) `market_value_meur` por equipo, ya presente en
        `files/f0_raw/teams_2026.csv` para las 48 selecciones (estos valores
        provienen, justamente, de Transfermarkt; ver README). Es un TOTAL de
        plantilla, sin desglose por jugador.
     b) Para repartir ese total por posición usamos una distribución canónica
        del valor de plantel (los delanteros concentran más valor que los
        laterales), de modo que el modelo aún diferencia ataque de defensa
        aunque no tenga el desglose real jugador a jugador.

Es decir: el camino por defecto NO es microsimulación jugador-a-jugador real;
es valor-de-mercado-de-plantel como PROXY de calidad, repartido por posición
con una plantilla típica. Cuando SÍ consigues datos por jugador (cache de
Transfermarkt rellenada a mano o scraping exitoso), el mismo pipeline los usa
sin cambios. La limitación está documentada también en `model.py` y el README.

Patrón de imports: igual que `core/data/*` — rutas absolutas desde la raíz del
repo, caché en `files/cache/`, sin dependencias nuevas obligatorias.
"""

from __future__ import annotations

import json
import time
import unicodedata
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TEAMS_CSV = ROOT / "files/f0_raw/teams_2026.csv"
TM_CACHE = ROOT / "files/cache/transfermarkt"

# Grupos de posición usados por todo el módulo (strength.py comparte estos).
POSITION_GROUPS = ("GK", "DEF", "MID", "ATT")

# Reparto canónico del VALOR de una plantilla de selección por grupo de
# posición (suma 1.0). Refleja que el mercado concentra el valor en el ataque
# y el mediocampo creativo, no en porteros/defensas. Se usa SOLO cuando no hay
# desglose real por jugador (fallback desde el total de `market_value_meur`).
# Fuente: distribución aproximada observada en plantillas top de Transfermarkt;
# es una HEURÍSTICA declarada, no un dato medido por equipo.
CANONICAL_VALUE_SHARE = {"GK": 0.06, "DEF": 0.26, "MID": 0.34, "ATT": 0.34}

# Plantilla típica (nº de jugadores por grupo) para repartir el valor del grupo
# en "jugadores sintéticos" cuando solo tenemos el total del equipo.
CANONICAL_SQUAD_SHAPE = {"GK": 3, "DEF": 8, "MID": 8, "ATT": 7}

# Mapa de posiciones crudas (Transfermarkt / FotMob) -> grupo canónico.
_RAW_POS_TO_GROUP = {
    "goalkeeper": "GK", "keeper": "GK", "gk": "GK", "tw": "GK",
    "centre-back": "DEF", "center-back": "DEF", "left-back": "DEF",
    "right-back": "DEF", "defender": "DEF", "full-back": "DEF", "cb": "DEF",
    "lb": "DEF", "rb": "DEF", "def": "DEF",
    "defensive midfield": "MID", "central midfield": "MID",
    "attacking midfield": "MID", "left midfield": "MID",
    "right midfield": "MID", "midfielder": "MID", "midfield": "MID",
    "mid": "MID", "dm": "MID", "cm": "MID", "am": "MID",
    "centre-forward": "ATT", "center-forward": "ATT", "second striker": "ATT",
    "left winger": "ATT", "right winger": "ATT", "forward": "ATT",
    "striker": "ATT", "winger": "ATT", "att": "ATT", "fw": "ATT",
}


def normalize_position(raw: str) -> str:
    """Mapea una posición cruda a uno de POSITION_GROUPS (default MID)."""
    if not raw:
        return "MID"
    key = str(raw).strip().lower()
    if key in _RAW_POS_TO_GROUP:
        return _RAW_POS_TO_GROUP[key]
    for token, group in _RAW_POS_TO_GROUP.items():
        if token in key:
            return group
    if key in POSITION_GROUPS:
        return key
    return "MID"


@dataclass
class PlayerValue:
    """Un jugador con su valor de mercado y su grupo de posición."""
    name: str
    position_group: str           # uno de POSITION_GROUPS
    value_meur: float             # valor de mercado en millones de EUR

    def __post_init__(self) -> None:
        if self.position_group not in POSITION_GROUPS:
            self.position_group = normalize_position(self.position_group)
        self.value_meur = max(0.0, float(self.value_meur))


@dataclass
class Squad:
    """Plantel de una selección, con la procedencia de los datos."""
    team: str
    players: list[PlayerValue]
    source: str                   # "transfermarkt" | "fotmob" | "synthetic_total"

    @property
    def total_value_meur(self) -> float:
        return sum(p.value_meur for p in self.players)

    def value_by_group(self) -> dict[str, float]:
        out = {g: 0.0 for g in POSITION_GROUPS}
        for p in self.players:
            out[p.position_group] += p.value_meur
        return out


def _norm_team(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return "_".join(s.lower().split())


# ----------------------------- fuentes en vivo -------------------------------
class RespectfulFetcher:
    """Cliente HTTP cortés: User-Agent real, rate-limiting global y back-off.

    No garantiza éxito: Transfermarkt/FotMob pueden devolver 403 o cambiar su
    HTML. Ante cualquier fallo devuelve None para que el orquestador degrade.
    """

    DEFAULT_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/124.0 Safari/537.36")

    def __init__(self, min_interval_s: float = 3.0, max_retries: int = 2,
                 timeout_s: float = 20.0, user_agent: Optional[str] = None):
        self.min_interval_s = min_interval_s
        self.max_retries = max_retries
        self.timeout_s = timeout_s
        self.user_agent = user_agent or self.DEFAULT_UA
        self._last_request_t = 0.0

    def get(self, url: str) -> Optional[str]:
        try:
            import requests
        except ImportError:
            return None
        for attempt in range(self.max_retries + 1):
            wait = self.min_interval_s - (time.monotonic() - self._last_request_t)
            if wait > 0:
                time.sleep(wait)
            try:
                resp = requests.get(
                    url, headers={"User-Agent": self.user_agent,
                                  "Accept-Language": "en,es;q=0.8"},
                    timeout=self.timeout_s)
                self._last_request_t = time.monotonic()
                if resp.status_code == 200:
                    return resp.text
                if resp.status_code in (429, 503):       # rate-limited: back-off
                    time.sleep(self.min_interval_s * (2 ** attempt))
                    continue
                return None                              # 403/404/etc.: no insistir
            except Exception:
                self._last_request_t = time.monotonic()
                time.sleep(self.min_interval_s * (2 ** attempt))
        return None


def parse_transfermarkt_squad(html: str, team: str) -> Optional[list[PlayerValue]]:
    """Extrae (jugador, posición, valor) de una página de plantilla de
    Transfermarkt. Best-effort: si el HTML no trae la tabla esperada, None."""
    try:
        from lxml import html as lxml_html
    except ImportError:
        return None
    try:
        doc = lxml_html.fromstring(html)
    except Exception:
        return None
    rows = doc.cssselect("table.items > tbody > tr")
    players: list[PlayerValue] = []
    for row in rows:
        name_el = row.cssselect("td.hauptlink a")
        pos_el = row.cssselect("td.posrela tr:last-child td")
        val_el = row.cssselect("td.rechts.hauptlink a")
        if not name_el or not val_el:
            continue
        name = name_el[0].text_content().strip()
        pos = pos_el[0].text_content().strip() if pos_el else "MID"
        val = _parse_money_to_meur(val_el[0].text_content())
        if val is None:
            continue
        players.append(PlayerValue(name, normalize_position(pos), val))
    return players or None


def _parse_money_to_meur(text: str) -> Optional[float]:
    """'€80.00m' / '€900k' / '$1.400.000' -> millones de EUR (aprox)."""
    if not text:
        return None
    t = str(text).strip().lower().replace(",", "").replace(" ", "")
    t = t.replace("€", "").replace("$", "").replace("£", "")
    try:
        if t.endswith("m"):
            return float(t[:-1])
        if t.endswith("k"):
            return float(t[:-1]) / 1000.0
        if t.endswith("bn"):
            return float(t[:-2]) * 1000.0
        # número crudo (p.ej. del dataset FIFA '1.400.000' ya sin comas)
        raw = float(t.replace(".", "")) if t.count(".") > 1 else float(t)
        return raw / 1e6 if raw > 1000 else raw
    except ValueError:
        return None


def fetch_transfermarkt_squad(team: str, url: str, *,
                              fetcher: Optional[RespectfulFetcher] = None
                              ) -> Optional[Squad]:
    """Scrapea la plantilla de `team` desde una URL de Transfermarkt.

    Necesitas pasar la URL de la página de plantilla (no se adivina el slug/id
    de cada selección, eso es frágil). Cachea el resultado en disco. Devuelve
    None si la petición o el parseo fallan — el orquestador hará fallback.
    """
    fetcher = fetcher or RespectfulFetcher()
    html = fetcher.get(url)
    if not html:
        return None
    players = parse_transfermarkt_squad(html, team)
    if not players:
        return None
    squad = Squad(team, players, source="transfermarkt")
    _write_cache(squad)
    return squad


def fetch_fotmob_squad(team: str, team_id: int, *,
                       fetcher: Optional[RespectfulFetcher] = None
                       ) -> Optional[Squad]:
    """Alternativa FotMob (API JSON interna `/api/teams?id=`).

    STUB HONESTO: FotMob no publica una API documentada y su payload cambia;
    además su ToS no autoriza scraping. Se deja el punto de entrada y el parseo
    mínimo del rating por jugador, pero por defecto NO se usa en producción.
    """
    fetcher = fetcher or RespectfulFetcher()
    raw = fetcher.get(f"https://www.fotmob.com/api/teams?id={team_id}")
    if not raw:
        return None
    try:
        data = json.loads(raw)
        squad_blocks = data.get("squad", [])
    except (json.JSONDecodeError, AttributeError):
        return None
    players: list[PlayerValue] = []
    for block in squad_blocks:
        role = block.get("title", "MID") if isinstance(block, dict) else "MID"
        for m in (block.get("members", []) if isinstance(block, dict) else []):
            # FotMob expone "rating" (0-10), no valor de mercado: lo escalamos a
            # un proxy de valor (rating^3) para que entre en el mismo pipeline.
            rating = m.get("rating")
            val = (float(rating) ** 3) if rating else 1.0
            players.append(PlayerValue(m.get("name", "?"),
                                       normalize_position(role), val))
    if not players:
        return None
    squad = Squad(team, players, source="fotmob")
    _write_cache(squad)
    return squad


# ------------------------------ caché en disco -------------------------------
def _cache_path(team: str) -> Path:
    return TM_CACHE / f"{_norm_team(team)}.json"


def _write_cache(squad: Squad) -> None:
    TM_CACHE.mkdir(parents=True, exist_ok=True)
    payload = {"team": squad.team, "source": squad.source,
               "players": [asdict(p) for p in squad.players]}
    _cache_path(squad.team).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_cached_squad(team: str) -> Optional[Squad]:
    path = _cache_path(team)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        players = [PlayerValue(**p) for p in payload["players"]]
        return Squad(payload["team"], players, payload.get("source", "cache"))
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


# --------------------------- fallback siempre-disponible ---------------------
def _team_total_value(team: str, teams_df: pd.DataFrame) -> Optional[float]:
    row = teams_df[teams_df["team"] == team]
    if row.empty:
        return None
    return float(row.iloc[0]["market_value_meur"])


def synthetic_squad_from_total(team: str, total_meur: float) -> Squad:
    """Construye un plantel SINTÉTICO repartiendo el valor total del equipo por
    posición según CANONICAL_VALUE_SHARE / CANONICAL_SQUAD_SHAPE.

    No son jugadores reales: son "ranuras" por posición que llevan su parte del
    valor del plantel. Permite que el modelo distinga ataque de defensa sin
    tener el desglose real. Declarado como `source='synthetic_total'`.
    """
    players: list[PlayerValue] = []
    for group in POSITION_GROUPS:
        group_value = total_meur * CANONICAL_VALUE_SHARE[group]
        n = CANONICAL_SQUAD_SHAPE[group]
        per_player = group_value / n if n else 0.0
        for i in range(n):
            players.append(PlayerValue(f"{team}_{group}{i+1}", group, per_player))
    return Squad(team, players, source="synthetic_total")


# ------------------------------- orquestador ---------------------------------
def load_squad(team: str, teams_df: pd.DataFrame, *,
               allow_network: bool = False,
               transfermarkt_urls: Optional[dict[str, str]] = None,
               fetcher: Optional[RespectfulFetcher] = None) -> Squad:
    """Devuelve el `Squad` de una selección con prioridad de fuentes:

    1. Caché en disco (`files/cache/transfermarkt/<team>.json`) — datos reales
       por jugador, si existen (scrapeados antes o rellenados a mano).
    2. Scraping en vivo de Transfermarkt SI `allow_network=True` y se dio una
       URL para el equipo en `transfermarkt_urls`. Respetuoso y cacheado.
    3. Fallback sintético desde `market_value_meur` (SIEMPRE disponible para
       las 48 selecciones). Honesto: no es por jugador, ver docstring del módulo.

    Nunca lanza por falta de red: degrada al fallback. Solo lanza si el equipo
    no existe en `teams_df` (error de datos, no de red).
    """
    cached = load_cached_squad(team)
    if cached is not None:
        return cached

    if allow_network and transfermarkt_urls and team in transfermarkt_urls:
        live = fetch_transfermarkt_squad(team, transfermarkt_urls[team],
                                         fetcher=fetcher)
        if live is not None:
            return live

    total = _team_total_value(team, teams_df)
    if total is None:
        raise ValueError(
            f"'{team}' no está en teams_2026.csv y no hay datos por jugador. "
            "No hay forma de estimar su valor de plantel.")
    return synthetic_squad_from_total(team, total)


def load_all_squads(teams_df: pd.DataFrame, **kwargs) -> dict[str, Squad]:
    """Carga el plantel de cada equipo del CSV (usa load_squad por equipo)."""
    return {t: load_squad(t, teams_df, **kwargs) for t in teams_df["team"]}
