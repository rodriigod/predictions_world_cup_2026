"""Extracción de features estructuradas por partido vía LLM + búsqueda web.

Contrato del encargo:
  - El LLM NO predice el resultado. Solo extrae SEÑALES OBJETIVAS Y
    VERIFICABLES, buscándolas en la web (no de su conocimiento interno, que
    puede estar desactualizado o alucinado).
  - Devuelve JSON estructurado (ver `schema.MatchFeatures`).
  - Caching agresivo (ver `cache.py`): no se repite la búsqueda del mismo
    partido entre corridas.
  - Si no hay información confiable para un campo -> `None`, nunca inventado.
  - Función pública: `get_match_features(team_home, team_away, match_date)`.

Proveedor de búsqueda web (dos arquitecturas, ambas = búsqueda web REAL)
-----------------------------------------------------------------------
1. **LM Studio (Qwen local) + DuckDuckGo** — DEFAULT. El LLM local NO tiene
   acceso a internet, así que NO se confía en su conocimiento interno: primero
   `ddg_match_search` trae snippets REALES de la web (DuckDuckGo, sin API key)
   y Qwen SOLO los lee para rellenar el JSON. Mismo patrón que
   `scripts/match_dossier.py`. Autodetecta el modelo cargado en LM Studio.
2. **Gemini grounded** — alternativa. `google_search` grounding hace la
   búsqueda web por su cuenta (free tier, `GEMINI_API_KEY`).

El proveedor es PLUGGABLE: cualquier `callable(prompt)->str` sirve (otro LLM,
un mock en tests). El buscador web (`searcher`) también es pluggable. Sin
proveedor / sin red -> degradación elegante a un esqueleto con todo en `None`.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from llm_features import cache
from llm_features.schema import MatchFeatures, TeamSignals

ROOT = Path(__file__).resolve().parents[1]
SEARCH_CACHE = ROOT / "files/cache/llm_features/search"

# Un proveedor es una función que recibe un prompt y devuelve el texto crudo
# del LLM (idealmente JSON), o None si la búsqueda falló.
Provider = Callable[[str], Optional[str]]
# Un buscador toma (home, away, date) y devuelve texto de contexto web (o "").
Searcher = Callable[[str, str, str], str]


# ----------------------------- carga de .env --------------------------------
def _load_dotenv(path: Path = ROOT / ".env") -> None:
    """Carga .env a os.environ sin depender de python-dotenv (igual que el repo)."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


# --------------------------- buscador: DuckDuckGo ---------------------------
def _team_news(team: str, *, max_results: int = 5, delay: float = 2.0) -> str:
    """Snippets reales de la web sobre UNA selección (DuckDuckGo, sin API key).
    Cachea por equipo en disco: una selección juega 3 partidos -> 1 sola búsqueda."""
    SEARCH_CACHE.mkdir(parents=True, exist_ok=True)
    path = SEARCH_CACHE / ("team_" + hashlib.md5(team.lower().encode()).hexdigest()
                           + ".txt")
    if path.exists():
        return path.read_text(encoding="utf-8")
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS
        q = (f"selección {team} fútbol Mundial 2026 lesiones bajas "
             "cambio entrenador forma reciente")
        with DDGS() as d:
            res = list(d.text(q, max_results=max_results))
        txt = " | ".join(f"{r.get('title','')}: {r.get('body','')}"
                         for r in res)[:1200]
    except Exception:
        return ""
    time.sleep(delay)
    if txt:
        path.write_text(txt, encoding="utf-8")
    return txt


def ddg_match_search(team_home: str, team_away: str, match_date: str, *,
                     max_results: int = 5, delay: float = 2.0) -> str:
    """Contexto web del partido = noticias reales de AMBAS selecciones (DDG)."""
    home = _team_news(team_home, max_results=max_results, delay=delay)
    away = _team_news(team_away, max_results=max_results, delay=delay)
    parts = []
    if home:
        parts.append(f"[{team_home}] {home}")
    if away:
        parts.append(f"[{team_away}] {away}")
    return "\n\n".join(parts)


# ------------------------- proveedor: LM Studio (Qwen) -----------------------
_LM_SYS = (
    "Eres un EXTRACTOR de datos deportivos. Lees un CONTEXTO de búsqueda web y "
    "rellenas un JSON con hechos. NO predices resultados, NO das opiniones, NO "
    "usas tu conocimiento previo: solo lo que esté en el contexto. Si un dato no "
    "aparece de forma confiable, lo dejas en null. Respondes SOLO el objeto JSON."
)


def lmstudio_provider(model: Optional[str] = None, url: Optional[str] = None,
                      api_key: Optional[str] = None, *, timeout_s: float = 120.0
                      ) -> Optional[Provider]:
    """Provider que usa LM Studio (endpoint OpenAI-compatible /chat/completions).

    Autodetecta el modelo cargado si el de `LMSTUDIO_MODEL` no está disponible.
    Devuelve None solo si no hay URL configurada; los fallos de red se manejan
    en cada llamada (devuelve None)."""
    _load_dotenv()
    url = url or os.environ.get("LMSTUDIO_URL", "http://localhost:1234/v1")
    if not url:
        return None
    api_key = api_key or os.environ.get("LMSTUDIO_API_KEY") or None
    model = model or os.environ.get("LMSTUDIO_MODEL")
    state = {"model": model, "resolved": False}

    def _headers() -> dict:
        h = {"Content-Type": "application/json"}
        if api_key:
            h["Authorization"] = f"Bearer {api_key}"
        return h

    def _resolve_model() -> Optional[str]:
        if state["resolved"]:
            return state["model"]
        state["resolved"] = True
        try:
            import requests
            r = requests.get(f"{url}/models", headers=_headers(), timeout=10)
            ids = [m["id"] for m in r.json().get("data", [])]
            chat = [i for i in ids if "embed" not in i.lower()]
            if (state["model"] not in ids) and chat:   # el configurado no está
                state["model"] = chat[0]               # usa el primero cargado
        except Exception:
            pass
        return state["model"]

    def _provider(prompt: str) -> Optional[str]:
        try:
            import requests
        except ImportError:
            return None
        m = _resolve_model()
        base = {"model": m, "temperature": 0, "max_tokens": 800,
                "messages": [{"role": "system", "content": _LM_SYS},
                             {"role": "user", "content": prompt}]}
        # 1º forzando objeto JSON; si LM Studio no lo soporta, libre
        for payload in ({**base, "response_format": {"type": "json_object"}}, base):
            try:
                r = requests.post(f"{url}/chat/completions", headers=_headers(),
                                  json=payload, timeout=timeout_s)
                if r.status_code != 200:
                    continue
                return r.json()["choices"][0]["message"]["content"]
            except Exception:
                continue
        return None

    return _provider


# --------------------------- proveedor: Gemini ------------------------------
def gemini_provider(model: Optional[str] = None, *, timeout_s: float = 45.0
                    ) -> Optional[Provider]:
    """Devuelve un Provider que usa Gemini + google_search, o None si no hay
    GEMINI_API_KEY. La búsqueda web la hace Gemini (grounding), no nosotros."""
    _load_dotenv()
    key = os.environ.get("GEMINI_API_KEY")
    if not key or key.startswith("your_"):
        return None
    model = model or os.environ.get("GEMINI_MODEL", "gemini-flash-latest")

    def _provider(prompt: str) -> Optional[str]:
        try:
            import requests
        except ImportError:
            return None
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
               f"{model}:generateContent")
        body = {"contents": [{"parts": [{"text": prompt}]}],
                "tools": [{"google_search": {}}]}
        try:
            r = requests.post(url, params={"key": key}, json=body,
                              timeout=timeout_s)
            if r.status_code != 200:
                return None
            data = r.json()
            parts = data["candidates"][0]["content"]["parts"]
            return "".join(p.get("text", "") for p in parts)
        except Exception:
            return None

    return _provider


def default_provider() -> Optional[Provider]:
    """Proveedor por defecto: LM Studio (local) si hay LMSTUDIO_URL, si no
    Gemini grounded si hay GEMINI_API_KEY, si no None."""
    _load_dotenv()
    if os.environ.get("LMSTUDIO_URL", "http://localhost:1234/v1"):
        lm = lmstudio_provider()
        if lm is not None:
            return lm
    return gemini_provider()


# ------------------------------- prompt -------------------------------------
_PROMPT_TEMPLATE = """\
Eres un asistente de EXTRACCIÓN DE DATOS deportivos. {source_instruction}
información ACTUAL y verificable sobre este partido del Mundial 2026:

  {team_home} (local) vs {team_away} (visitante) — fecha {match_date}
{web_context}
NO predigas el resultado ni des opiniones. Devuelve SOLO hechos buscables.
Si NO encuentras información confiable para un campo, ponlo en null. Nunca
inventes nombres, números ni porcentajes.

Devuelve EXCLUSIVAMENTE un objeto JSON válido (sin texto alrededor, sin ```)
con esta forma EXACTA:

{{
  "home": {{
    "lesionados_clave": [lista de nombres de titulares/figuras de {team_home}
        confirmados FUERA por lesión, o null si no hay/ no se sabe],
    "cambio_dt_reciente": true/false/null (¿cambió de entrenador en los
        últimos ~3 meses?),
    "consenso_expertos_pct": número 0-100 o null (% de analistas/medios
        reconocidos que, según predicciones PUBLICADAS reales, favorecen a
        {team_home}),
    "fatiga_viaje_km": número o null (km recorridos desde su último partido),
    "fatiga_husos_horarios": entero o null (husos horarios cruzados)
  }},
  "away": {{ ...mismos campos para {team_away}... }},
  "dead_rubber": true/false/null (¿es un partido de fase de grupos
      intrascendente porque {team_home} o {team_away} YA está clasificado o
      eliminado matemáticamente antes de jugar?),
  "notes": "1-2 frases con las fuentes/citas en que te basaste, o null"
}}
"""


def build_prompt(team_home: str, team_away: str, match_date: str,
                 web_context: Optional[str] = None) -> str:
    """Construye el prompt. Si se pasa `web_context` (snippets de DDG), el LLM
    debe usar ESO como única fuente (modo LM Studio local); si no, se le pide
    que BUSQUE en la web por su cuenta (modo Gemini grounded)."""
    if web_context:
        source_instruction = ("Basándote ÚNICAMENTE en el CONTEXTO DE BÚSQUEDA "
                              "WEB de abajo (no uses tu conocimiento previo), "
                              "extrae")
        ctx = (f"\nCONTEXTO DE BÚSQUEDA WEB (única fuente válida):\n"
               f"\"\"\"\n{web_context}\n\"\"\"\n")
    else:
        source_instruction = "Usa BÚSQUEDA WEB para encontrar"
        ctx = ""
    return _PROMPT_TEMPLATE.format(
        team_home=team_home, team_away=team_away, match_date=match_date,
        source_instruction=source_instruction, web_context=ctx)


# ----------------------------- parseo robusto -------------------------------
def _extract_json(text: str) -> Optional[dict]:
    """Saca el primer objeto JSON del texto del LLM (tolera ``` y prosa)."""
    if not text:
        return None
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(cleaned[start:end + 1])
    except json.JSONDecodeError:
        return None


def _as_float(v) -> Optional[float]:
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def _as_int(v) -> Optional[int]:
    f = _as_float(v)
    return None if f is None else int(round(f))


def _as_bool(v) -> Optional[bool]:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        if v.strip().lower() in ("true", "sí", "si", "yes"):
            return True
        if v.strip().lower() in ("false", "no"):
            return False
    return None


def _as_str_list(v) -> Optional[list[str]]:
    if v is None:
        return None
    if isinstance(v, str):
        v = [v]
    if not isinstance(v, (list, tuple)):
        return None
    _junk = {"null", "none", "n/a", "na", "ninguno", "ninguna", "-", "?"}
    out = [str(x).strip() for x in v
           if str(x).strip() and str(x).strip().lower() not in _junk]
    return out or None


def _parse_team(d: dict) -> TeamSignals:
    d = d or {}
    pct = _as_float(d.get("consenso_expertos_pct"))
    if pct is not None:
        pct = float(min(100.0, max(0.0, pct)))     # clamp a [0,100]
    return TeamSignals(
        lesionados_clave=_as_str_list(d.get("lesionados_clave")),
        cambio_dt_reciente=_as_bool(d.get("cambio_dt_reciente")),
        consenso_expertos_pct=pct,
        fatiga_viaje_km=_as_float(d.get("fatiga_viaje_km")),
        fatiga_husos_horarios=_as_int(d.get("fatiga_husos_horarios")),
    )


def parse_features(raw: str, team_home: str, team_away: str, match_date: str,
                   provider_name: Optional[str] = None) -> Optional[MatchFeatures]:
    """Convierte el texto crudo del LLM en MatchFeatures, o None si no parsea.
    Nunca inventa: lo que no venga o no parsee queda en None."""
    data = _extract_json(raw)
    if data is None:
        return None
    notes = data.get("notes")
    return MatchFeatures(
        team_home=team_home, team_away=team_away, match_date=str(match_date),
        home=_parse_team(data.get("home")),
        away=_parse_team(data.get("away")),
        dead_rubber=_as_bool(data.get("dead_rubber")),
        source="llm_web_search",
        retrieved_at=datetime.now(timezone.utc).isoformat(),
        provider=provider_name,
        notes=(str(notes) if notes else None),
    )


# ------------------------------ API pública ---------------------------------
_DEFAULT_SEARCHER = object()   # centinela: "usa ddg_match_search por defecto"


def get_match_features(team_home: str, team_away: str, match_date: str, *,
                       provider: Optional[Provider] = None,
                       searcher=_DEFAULT_SEARCHER,
                       use_cache: bool = True, allow_network: bool = True,
                       max_age_days: Optional[float] = None,
                       provider_name: Optional[str] = None) -> dict:
    """Devuelve el dict de features estructuradas del partido (schema anidado).

    Orden: caché -> búsqueda web + LLM (si hay proveedor y allow_network) ->
    esqueleto null. NUNCA lanza por falta de red/clave: degrada a todo None.

    - `provider`: callable(prompt)->str. None -> `default_provider()` (LM Studio
      local si hay LMSTUDIO_URL; si no Gemini grounded; si no, sin extracción).
    - `searcher`: callable(home,away,date)->str que trae contexto web. Por
      defecto `ddg_match_search` (DuckDuckGo). Pásalo en None para que el
      proveedor busque solo (p.ej. Gemini grounded), o un mock en tests.
    - `use_cache`: lee/escribe `files/cache/llm_features/`. `max_age_days`
      fuerza refresco si la caché es más vieja que ese umbral.

    Devuelve `MatchFeatures.to_dict()`. Para columnas planas del ensemble usa
    `MatchFeatures.from_dict(result).to_flat_dict()`.
    """
    if use_cache:
        cached = cache.load(team_home, team_away, match_date,
                            max_age_days=max_age_days)
        if cached is not None:
            return cached.to_dict()

    if not allow_network:
        return MatchFeatures.null(team_home, team_away, match_date).to_dict()

    prov = provider if provider is not None else default_provider()
    if prov is None:                                  # sin proveedor disponible
        return MatchFeatures.null(team_home, team_away, match_date).to_dict()
    if provider is None and provider_name is None:    # trazabilidad del default
        provider_name = ("lmstudio"
                         if os.environ.get("LMSTUDIO_URL", "http://localhost:1234/v1")
                         else "gemini")

    # contexto web real (DuckDuckGo): la "única fuente" para el LLM local
    search_fn = ddg_match_search if searcher is _DEFAULT_SEARCHER else searcher
    web_context = None
    if search_fn is not None:
        try:
            web_context = search_fn(team_home, team_away, match_date) or None
        except Exception:
            web_context = None

    raw = prov(build_prompt(team_home, team_away, match_date, web_context))
    feats = (parse_features(raw, team_home, team_away, match_date,
                            provider_name=provider_name)
             if raw else None)
    if feats is None:                                 # red OK pero sin datos útiles
        return MatchFeatures.null(team_home, team_away, match_date).to_dict()

    if use_cache:
        cache.save(feats)
    return feats.to_dict()


def get_features_for_fixtures(fixtures, **kwargs) -> list[dict]:
    """Extrae features para todos los partidos de un fixture (cols team_a,
    team_b, date). Reusa caché entre partidos automáticamente."""
    out = []
    for fx in fixtures.itertuples(index=False):
        d = fx._asdict()
        out.append(get_match_features(str(d["team_a"]), str(d["team_b"]),
                                      str(d["date"]), **kwargs))
    return out
