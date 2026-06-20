#!/usr/bin/env python3
"""Dossier por partido: modelo numérico + noticias (Gemini) + análisis (Qwen local).

Capa de SCOUTING sobre el forecast calibrado, NO un reemplazo. Para cada partido:
  1) Gemini (free tier, GEMINI_API_KEY) busca info blanda: lesiones, alineación,
     bajas, rotación, contexto reciente.
  2) Un LLM local (Qwen vía LM Studio, endpoint OpenAI-compatible en
     localhost:1234) lee las probabilidades del modelo + esas noticias y propone
     un AJUSTE ACOTADO (±DELTA_MAX por clase, justificado). El número del modelo
     manda; la IA solo matiza.
  3) Se escribe una tabla tipo README: marcador + P(1/X/2) modelo + P ajustada + nota.

Degradación elegante: sin GEMINI_API_KEY no hay noticias; sin LM Studio no hay
ajuste (ajuste=0). En ambos casos la tabla se genera igual (modelo solo).

Uso:
  export GEMINI_API_KEY=...
  # arranca LM Studio con un modelo Qwen cargado (Local Server -> Start)
  python scripts/match_dossier.py --limit 8
  python scripts/match_dossier.py --analyzer none   # solo modelo, sin LLM (testeable offline)
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import requests

PRED = ROOT / "files/f3_output/match_predictions.csv"
OUT = ROOT / "results/reports/dossier.md"
DELTA_MAX = 0.08   # ajuste máximo por clase que la IA puede aplicar (acota su poder)


def _load_dotenv(path: Path) -> None:
    """Carga .env a os.environ (sin depender de python-dotenv)."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


# --------------------------- etapa 1: Gemini (por EQUIPO real) ---------------
import hashlib
import time

GEM_CACHE = ROOT / "files/cache/gemini"


def gemini_team_news(team: str, key: str, model: str, delay: float = 4.0) -> str:
    """Info ACTUAL del EQUIPO real (forma, figuras, lesiones) vía Gemini + web.
    Pregunta por el equipo (real), NO por el cruce de la polla (puede ser
    ficticio). Cachea en disco: un equipo juega 3 partidos -> 1 sola consulta."""
    GEM_CACHE.mkdir(parents=True, exist_ok=True)
    cache = GEM_CACHE / (hashlib.md5(team.lower().encode()).hexdigest() + ".txt")
    if cache.exists():
        return cache.read_text(encoding="utf-8")
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model}:generateContent")
    prompt = (f"Busca en la web información ACTUAL (2026) de la selección de fútbol "
              f"de {team}: forma reciente (últimos resultados), 1-2 jugadores clave "
              "y cualquier lesión o baja importante. Máximo 2 frases, solo hechos. "
              "Si no encuentras nada, responde exactamente 'sin datos'.")
    body = {"contents": [{"parts": [{"text": prompt}]}],
            "tools": [{"google_search": {}}]}
    for payload in (body, {"contents": body["contents"]}):   # reintento sin tool
        try:
            r = requests.post(url, params={"key": key}, json=payload, timeout=40)
            if r.status_code == 429:
                print(f"    ⚠️ Gemini 429 (cuota) en {team}; espero {delay*3:.0f}s...")
                time.sleep(delay * 3)
                r = requests.post(url, params={"key": key}, json=payload, timeout=40)
            if r.status_code != 200:
                print(f"    ⚠️ Gemini HTTP {r.status_code} en {team}")
                continue
            cand = r.json()["candidates"][0]["content"]["parts"]
            txt = " ".join(p.get("text", "") for p in cand).strip()
            time.sleep(delay)            # respeta el rate limit del free tier
            if txt:                      # NO cachear vacíos (envenenaría la caché)
                cache.write_text(txt, encoding="utf-8")
                return txt
        except Exception as e:
            print(f"    ⚠️ Gemini error en {team}: {type(e).__name__}")
            continue
    return ""


# --------------------------- etapa 1 (alt): DuckDuckGo (sin key, gratis) -----
def ddg_team_news(team: str, delay: float = 3.0) -> str:
    """Busca info real del EQUIPO en la web vía DuckDuckGo (sin API key ni cuota).
    Concatena los snippets de los primeros resultados. Cachea en disco."""
    GEM_CACHE.mkdir(parents=True, exist_ok=True)
    cache = GEM_CACHE / ("ddg_" + hashlib.md5(team.lower().encode()).hexdigest() + ".txt")
    if cache.exists():
        return cache.read_text(encoding="utf-8")
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS
        q = f"selección {team} fútbol forma reciente lesiones bajas 2026"
        with DDGS() as d:
            res = list(d.text(q, max_results=5))
        txt = " | ".join(f"{r.get('title','')}: {r.get('body','')}" for r in res)[:900]
    except Exception as e:
        print(f"    ⚠️ DuckDuckGo error en {team}: {type(e).__name__}")
        return ""
    time.sleep(delay)
    if txt:
        cache.write_text(txt, encoding="utf-8")
        return txt
    return ""


# --------------------------- etapa 2: LLM local ---------------------------
SYS = ("Eres un analista de fútbol. Te doy las probabilidades 1X2 de un modelo "
       "calibrado y noticias recientes. El modelo es la base; tú solo ajustas "
       f"levemente si las noticias lo justifican (máx ±{DELTA_MAX:.2f} por clase). "
       "Devuelve EXCLUSIVAMENTE un objeto JSON, sin texto antes ni después, sin "
       "markdown, con esta forma exacta: "
       '{"dh": 0.0, "dd": 0.0, "da": 0.0, "reason": "máx 12 palabras"}. '
       "dh=ajuste a P(gana local), dd=empate, da=gana visita. "
       "Sin noticias relevantes, los tres deltas son 0.")

# JSON Schema para structured output de LM Studio (fuerza salida JSON válida)
_RF = {"type": "json_schema", "json_schema": {"name": "adjust", "strict": True,
       "schema": {"type": "object", "additionalProperties": False,
                  "properties": {"dh": {"type": "number"}, "dd": {"type": "number"},
                                 "da": {"type": "number"}, "reason": {"type": "string"}},
                  "required": ["dh", "dd", "da", "reason"]}}}


def _extract_json(txt: str):
    """Saca el JSON de la respuesta del LLM aunque venga con <think>, ```fences``` o
    texto alrededor. Devuelve dict o None."""
    txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.DOTALL)   # modelos razonadores
    txt = txt.replace("```json", "```")
    for cand in reversed(re.findall(r"\{[^{}]*\}", txt, re.DOTALL)):  # objetos simples
        try:
            return json.loads(cand)
        except Exception:
            pass
    m = re.search(r"\{.*\}", txt, re.DOTALL)                          # primer { a último }
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


def llm_adjust(home, away, p, news, url, model, api_key=None):
    """Pide al LLM local un ajuste 1X2 acotado. Devuelve (dh,dd,da,reason).
    Fuerza JSON con structured output; si el modelo/LM Studio no lo soporta,
    reintenta sin él y parsea de forma robusta."""
    user = (f"{home} vs {away}\nModelo P(gana {home})={p[0]:.2f} "
            f"P(empate)={p[1]:.2f} P(gana {away})={p[2]:.2f}\n"
            f"Noticias: {news or '(sin noticias)'}")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    base = {"model": model, "temperature": 0, "max_tokens": 300,
            "messages": [{"role": "system", "content": SYS},
                         {"role": "user", "content": user}]}
    note = "(LLM sin JSON)"
    for payload in ({**base, "response_format": _RF}, base):   # 1º forzado, luego libre
        try:
            r = requests.post(f"{url}/chat/completions", timeout=120,
                              headers=headers, json=payload)
            if r.status_code != 200:
                note = f"(LLM HTTP {r.status_code})"
                continue
            txt = r.json()["choices"][0]["message"]["content"]
            d = _extract_json(txt)
            if d is None:
                continue
            return (float(d.get("dh", 0) or 0), float(d.get("dd", 0) or 0),
                    float(d.get("da", 0) or 0), str(d.get("reason", ""))[:60] or "ok")
        except Exception as e:
            note = f"(LLM no disponible: {type(e).__name__})"
    return (0.0, 0.0, 0.0, note)


def apply_adjustment(p, deltas):
    """Aplica deltas acotados y renormaliza (el modelo sigue siendo el ancla)."""
    import numpy as np
    d = np.clip(np.array(deltas[:3]), -DELTA_MAX, DELTA_MAX)
    out = np.clip(np.array(p) + d, 1e-4, None)
    return (out / out.sum()).tolist()


# --------------------------- pipeline ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyzer", choices=["local", "none"], default="local",
                    help="local = Qwen vía LM Studio; none = solo modelo (offline)")
    ap.add_argument("--search", choices=["ddg", "gemini", "none"], default="ddg",
                    help="buscador de info por equipo: ddg = DuckDuckGo (gratis, "
                         "sin key, default); gemini = Gemini (necesita key+cuota); "
                         "none = sin búsqueda")
    ap.add_argument("--lmstudio-url", default="http://localhost:1234/v1")
    ap.add_argument("--model", default="local-model",
                    help="id del modelo cargado en LM Studio (suele dar igual)")
    ap.add_argument("--gemini-model", default="gemini-flash-latest")
    ap.add_argument("--delay", type=float, default=3.0,
                    help="segundos entre búsquedas (sube a 6 si gemini te da 429)")
    ap.add_argument("--limit", type=int, default=0, help="0 = todos los pendientes")
    ap.add_argument("--all", action="store_true", help="incluir partidos jugados")
    args = ap.parse_args()
    _load_dotenv(ROOT / ".env")
    key = os.environ.get("GEMINI_API_KEY")
    lm_url = os.environ.get("LMSTUDIO_URL", args.lmstudio_url)
    lm_key = os.environ.get("LMSTUDIO_API_KEY")
    lm_model = os.environ.get("LMSTUDIO_MODEL", args.model)
    gem_model = os.environ.get("GEMINI_MODEL", args.gemini_model)

    df = pd.read_csv(PRED)
    if not args.all:
        df = df[df["status"] != "JUGADO"]
    if args.limit:
        df = df.head(args.limit)
    use_llm = args.analyzer == "local"
    search = args.search if (args.search != "gemini" or key) else "none"
    print(f"Dossier de {len(df)} partidos | búsqueda: {search} | "
          f"LLM local: {'sí' if use_llm else 'no'} ({lm_url})")

    def fetch(team):
        if search == "ddg":
            return ddg_team_news(team, args.delay)
        if search == "gemini":
            return gemini_team_news(team, key, gem_model, args.delay)
        return ""

    rows = []
    team_news = {}   # equipo -> info real (para la sección de noticias, sin repetir)
    for r in df.itertuples():
        p = [float(r.p_win_a), float(r.p_draw), float(r.p_win_b)]
        if search != "none":
            for t in (r.team_a, r.team_b):
                if t not in team_news:
                    team_news[t] = fetch(t)
            na, nb = team_news[r.team_a], team_news[r.team_b]
            news = f"{r.team_a}: {na or 'sin datos'} || {r.team_b}: {nb or 'sin datos'}"
        else:
            news = ""
        if use_llm:
            dh, dd, da, reason = llm_adjust(r.team_a, r.team_b, p, news,
                                            lm_url, lm_model, lm_key)
        else:
            dh = dd = da = 0.0
            reason = "(solo modelo)"
        adj = apply_adjustment(p, (dh, dd, da))
        rows.append((r.group, r.team_a, r.team_b, r.pred_score, p, adj, reason,
                     float(r.exp_goals_a), float(r.exp_goals_b)))
        print(f"  {r.team_a} vs {r.team_b}: {reason}")

    # ---- tabla tipo README ----
    out = [f"# 🔮 Dossier por partido — modelo + búsqueda web ({search}) + Qwen\n",
           "Capa de scouting sobre el forecast calibrado. El número del modelo es "
           f"el ancla; la IA solo ajusta ±{DELTA_MAX:.2f} por clase si hay noticias "
           "que lo justifiquen. **No reemplaza** la predicción base.\n",
           "| Grp | Partido | Marcador | Goles esp. | P(1)/X/P(2) modelo "
           "| P(1)/X/P(2) ajustada | Nota |",
           "|:-:|---|:-:|:-:|:-:|:-:|---|"]
    for g, a, b, sc, p, adj, reason, ega, egb in rows:
        pm = f"{p[0]*100:.0f}/{p[1]*100:.0f}/{p[2]*100:.0f}"
        pa = f"{adj[0]*100:.0f}/{adj[1]*100:.0f}/{adj[2]*100:.0f}"
        flag = "" if pm == pa else " ⚙️"
        out.append(f"| {g} | {a} – {b} | {sc} | {ega:.2f}-{egb:.2f} "
                   f"| {pm} | {pa}{flag} | {reason} |")
    out.append("\n⚙️ = la IA ajustó por noticias. Probabilidades en % (gana primero "
               "/ empate / gana segundo).")
    real_news = {t: n for t, n in team_news.items() if n and n.lower() != "sin datos"}
    if real_news:
        out.append("\n## 📰 Info por equipo (búsqueda web)\n")
        for t, n in real_news.items():
            out.append(f"- **{t}**: {n}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(out), encoding="utf-8")
    print(f"\nTabla -> {OUT}")


if __name__ == "__main__":
    main()
