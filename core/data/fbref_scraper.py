"""Scraper minimalista de FBref para stats por jugador (motor pre-partido).

Stats objetivo (por 90', última temporada de club con ≥500 min):
  npxg_p90, progressive_carries_p90, pressures_p90, tackles_p90,
  aerials_won_pct, psxg_save_pct (porteros)

ADVERTENCIAS HONESTAS:
- FBref (Sports Reference) bloquea scrapers de forma agresiva (HTTP 403/429) y
  pide ~6s entre requests. Si bloquea, esta función devuelve {} y el motor usa
  los promedios por posición — el resultado sigue siendo válido pero NO refleja
  la calidad real del jugador.
- FBref ESCONDE la mayoría de sus tablas dentro de comentarios HTML para frenar
  bots: aquí se des-comentan antes de parsear.
- La stat `pressures` fue DISCONTINUADA por FBref (era de StatsBomb). Casi nunca
  estará disponible → cae al default de posición. Es esperado.

Cachea en files/cache/fbref/ para no repetir búsquedas.
"""
import hashlib
import json
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup, Comment

CACHE_DIR = Path(__file__).resolve().parents[2] / "files/cache/fbref"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}

# nombre FBref (data-stat) -> nombre interno
STAT_MAP = {
    "npxg_per90": "npxg_p90",
    "progressive_carries": "progressive_carries_p90",  # se normaliza por 90 abajo
    "carries_progressive": "progressive_carries_p90",
    "pressures_per90": "pressures_p90",
    "tackles_per90": "tackles_p90",
    "tackles": "tackles_p90",
    "aerials_won_pct": "aerials_won_pct",
    "aerials_won_pct_keeper": "aerials_won_pct",
    "gk_psxg_save_pct": "psxg_save_pct",
    "gk_psnpxg_per_shot_on_target_against": None,
}


def _cache_path(name: str) -> Path:
    return CACHE_DIR / f"{hashlib.md5(name.lower().encode()).hexdigest()}.json"


def _get(url: str, timeout: int = 15):
    return requests.get(url, headers=HEADERS, timeout=timeout)


def _soup_with_comments(html: str) -> BeautifulSoup:
    """FBref mete tablas dentro de <!-- comentarios -->. Los des-comentamos
    para que BeautifulSoup los vea como HTML real."""
    soup = BeautifulSoup(html, "html.parser")
    for c in soup.find_all(string=lambda t: isinstance(t, Comment)):
        if "<table" in c:
            frag = BeautifulSoup(c, "html.parser")
            c.replace_with(frag)
    return soup


def _to_float(text: str):
    text = (text or "").strip().replace(",", "")
    if not text:
        return None
    pct = "%" in text
    try:
        v = float(text.replace("%", ""))
    except ValueError:
        return None
    return v / 100.0 if pct else v


def search_player_fbref(player_name: str, delay: float = 6.0) -> dict:
    """Busca el jugador en FBref y devuelve sus stats (cacheado). {} si falla."""
    cp = _cache_path(player_name)
    if cp.exists():
        return json.loads(cp.read_text())

    try:
        time.sleep(delay)
        q = requests.utils.quote(player_name)
        resp = _get(f"https://fbref.com/search/search.fcgi?search={q}")
        if resp.status_code != 200:
            # 403/429 = bloqueo TRANSITORIO: NO cachear (si no, quedaría
            # "sin datos" para siempre). Solo se cachea un parseo exitoso.
            print(f"  ⚠ búsqueda falló {player_name}: HTTP {resp.status_code} "
                  "(FBref bloquea scrapers; usa defaults o stats manuales)")
            return {}

        # la búsqueda puede redirigir directo a la página del jugador
        if "/players/" in resp.url and "/search/" not in resp.url:
            player_url = resp.url
        else:
            soup = BeautifulSoup(resp.text, "html.parser")
            player_url = None
            for a in soup.select("div.search-item-name a, .search-item a"):
                href = a.get("href", "")
                if "/players/" in href:
                    player_url = "https://fbref.com" + href if href.startswith("/") else href
                    break
            if not player_url:
                print(f"  ⚠ sin página FBref para: {player_name}")
                cp.write_text("{}"); return {}

        time.sleep(delay)
        pr = _get(player_url)
        if pr.status_code != 200:
            print(f"  ⚠ página falló {player_name}: HTTP {pr.status_code}")
            return {}   # bloqueo transitorio: no cachear

        stats = _parse_player_page(_soup_with_comments(pr.text))
        stats["fbref_url"] = player_url
        stats["player_name"] = player_name
        cp.write_text(json.dumps(stats, indent=2))
        got = {k: v for k, v in stats.items()
               if k in ("npxg_p90", "pressures_p90", "psxg_save_pct")}
        print(f"  ✓ {player_name}: {got or 'sin stats útiles (usa defaults)'}")
        return stats
    except Exception as e:
        print(f"  ✗ error {player_name}: {e}")
        return {}   # error de red: no cachear


def _parse_player_page(soup: BeautifulSoup) -> dict:
    """Extrae stats de la temporada de liga más reciente con ≥500 min."""
    stats: dict = {}
    tables = [t for t in soup.find_all("table")
              if (t.get("id") or "").startswith("stats_")]
    for table in tables:
        body = table.find("tbody")
        if not body:
            continue
        for row in reversed(body.find_all("tr")):
            if "thead" in (row.get("class") or []):
                continue
            mins = row.find("td", {"data-stat": "minutes"})
            mv = _to_float(mins.text) if mins else None
            if mv is None or mv < 500:
                continue
            n90 = max(mv / 90.0, 1e-6)
            for ds, internal in STAT_MAP.items():
                if internal is None:
                    continue
                cell = row.find("td", {"data-stat": ds})
                if not cell or not cell.text.strip():
                    continue
                val = _to_float(cell.text)
                if val is None:
                    continue
                # contadores crudos (no _per90 ni _pct) -> normalizar por 90
                if ds in ("progressive_carries", "tackles"):
                    val = val / n90
                stats.setdefault(internal, val)
            break  # primera (más reciente) temporada válida de esta tabla
    return stats


STAT_FIELDS = ("npxg_p90", "progressive_carries_p90", "pressures_p90",
               "tackles_p90", "aerials_won_pct", "psxg_save_pct")


def get_squad_stats(lineup: list, delay: float = 6.0) -> list:
    """Devuelve [Player] para la alineación. Prioridad de stats:
    1) las que vengan EN EL JSON del jugador (manuales, evitan el scraper),
    2) FBref (si responde — suele dar 403),
    3) defaults por posición (fallback)."""
    from core.simulation.match_engine import Player
    n_manual = sum(1 for p in lineup if any(k in p for k in STAT_FIELDS))
    print(f"\nStats de {len(lineup)} jugadores ({n_manual} manuales en el JSON, "
          f"resto vía FBref ~{delay:.0f}s c/u la 1ª vez)...")
    players = []
    for p in lineup:
        manual = {k: p[k] for k in STAT_FIELDS if k in p}
        raw = manual if manual else search_player_fbref(p["name"], delay=delay)
        players.append(Player(
            name=p["name"], position=p["position"],
            npxg_p90=raw.get("npxg_p90", 0.0),
            progressive_carries_p90=raw.get("progressive_carries_p90", 0.0),
            pressures_p90=raw.get("pressures_p90", 0.0),
            tackles_p90=raw.get("tackles_p90", 0.0),
            aerials_won_pct=raw.get("aerials_won_pct", 0.50),
            psxg_save_pct=raw.get("psxg_save_pct", 0.72)))
    return players
