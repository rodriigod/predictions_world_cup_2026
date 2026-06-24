#!/usr/bin/env python3
"""B. CLV tracking — captura de cuota de apertura/cierre y cálculo de CLV.

Registra, por partido y en `results/clv_tracking.csv`:
  fecha, equipos, tu_prob, cuota_apertura, cuota_cierre, clv_resultante (+ extras).

Flujo (pre-registro honesto, igual que el log de amistosos):
  1. AL PREDECIR: se guarda tu_prob + la cuota de APERTURA (open/early line).
  2. JUSTO ANTES DEL PARTIDO: se añade la cuota de CIERRE y se calcula el CLV,
     SIN tocar la apertura ya guardada.

Fuente de cuotas: **The Odds API** (the-odds-api.com), tier gratuito, vía la
variable de entorno `ODDS_API_KEY`. SIN clave, las funciones de red devuelven
None y el registro se hace con las cuotas que se pasen a mano (o queda pendiente):
NO se inventan cuotas. Limitaciones documentadas en
`results/reports/clv_tracking.md`.

Uso:
  # con clave en .env (ODDS_API_KEY=...):
  python scripts/clv_tracking.py --open "España" "Alemania" 2026-03-25
  python scripts/clv_tracking.py --close "España" "Alemania" 2026-03-25
  # cuotas a mano (decimal, 1 X 2):
  python scripts/clv_tracking.py --open  "España" "Alemania" 2026-03-25 --odds 2.10 3.40 3.50
  python scripts/clv_tracking.py --close "España" "Alemania" 2026-03-25 --odds 1.95 3.50 4.00
  python scripts/clv_tracking.py --summary
"""

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from core.data.clv import OUTCOMES, compute_clv, summarize_clv
from core.data.odds_tools import demargin, logit_consensus

CSV = ROOT / "results/clv_tracking.csv"
REPORT = ROOT / "results/reports/clv_tracking.md"

COLS = ["match_id", "fecha", "home", "away", "equipos", "pick", "tu_prob",
        "cuota_apertura", "cuota_cierre", "clv_resultante", "edge_vs_cierre",
        "open_1", "open_X", "open_2", "close_1", "close_X", "close_2",
        "odds_source", "open_at", "close_at"]

# deportes de The Odds API relevantes (cobertura variable según fecha/temporada)
ODDS_SPORTS = ["soccer_fifa_world_cup", "soccer_uefa_nations_league",
               "soccer_international_friendlies"]


def _mid(home, away, date) -> str:
    return f"{str(date).strip()}|{home.strip()}|{away.strip()}"


def _load() -> pd.DataFrame:
    if CSV.exists():
        return pd.read_csv(CSV, dtype={"match_id": str})
    return pd.DataFrame(columns=COLS)


def _save(df: pd.DataFrame) -> None:
    CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV, index=False)


# --------------------------- fuente de cuotas (red) -------------------------
def _load_env() -> None:
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def fetch_odds_1x2(home: str, away: str, date: str):
    """Cuota 1X2 decimal de consenso (des-marginada por casa, promediada en log)
    desde The Odds API, o None si no hay clave / no hay cobertura / falla la red.

    Devuelve (odds_1x2, source) con odds_1x2 = cuotas decimales JUSTAS de consenso
    reconstruidas (1/prob), o None. NO inventa: si no encuentra el partido, None.
    """
    _load_env()
    key = os.environ.get("ODDS_API_KEY")
    if not key or key.startswith("your_"):
        return None
    try:
        import requests
    except ImportError:
        return None
    from core.data.historical import NAME_MAP
    h_en = NAME_MAP.get(home, home).lower()
    a_en = NAME_MAP.get(away, away).lower()
    for sport in ODDS_SPORTS:
        try:
            r = requests.get(
                f"https://api.the-odds-api.com/v4/sports/{sport}/odds",
                params={"apiKey": key, "regions": "eu,uk", "markets": "h2h",
                        "oddsFormat": "decimal"}, timeout=20)
            if r.status_code != 200:
                continue
            for ev in r.json():
                names = {ev.get("home_team", "").lower(),
                         ev.get("away_team", "").lower()}
                if not ({h_en, a_en} <= names):
                    continue
                rows = []
                for bk in ev.get("bookmakers", []):
                    o = _parse_h2h(bk, ev["home_team"], ev["away_team"])
                    if o is not None:
                        rows.append(demargin(o, method="shin"))
                if rows:
                    cons = logit_consensus(rows)          # prob consenso [1,X,2]
                    odds = [round(1.0 / max(p, 1e-9), 3) for p in cons]
                    return odds, f"theoddsapi:{sport}:{len(rows)}books"
        except Exception:
            continue
    return None


def _parse_h2h(bookmaker, home_team, away_team):
    for mk in bookmaker.get("markets", []):
        if mk.get("key") != "h2h":
            continue
        price = {o["name"]: o["price"] for o in mk.get("outcomes", [])}
        if home_team in price and away_team in price and "Draw" in price:
            return [price[home_team], price["Draw"], price[away_team]]
    return None


# ------------------------------ registro ------------------------------------
def record_open(home, away, date, model_probs, odds_1x2=None, source=None) -> dict:
    """Guarda tu_prob + cuota de APERTURA. Idempotente (no re-escribe si existe)."""
    df = _load()
    mid = _mid(home, away, date)
    if (df["match_id"] == mid).any():
        print(f"  ya existe apertura: {mid}")
        return df[df["match_id"] == mid].iloc[0].to_dict()
    if odds_1x2 is None:
        fetched = fetch_odds_1x2(home, away, date)
        if fetched is None:
            print(f"  ⚠️ sin cuotas para {mid} (sin ODDS_API_KEY o sin cobertura); "
                  "se registra tu_prob, cuota de apertura pendiente")
            odds_1x2, source = None, "pending"
        else:
            odds_1x2, source = fetched
    else:
        source = source or "manual"

    pick = int(np.argmax(model_probs))
    row = {c: None for c in COLS}
    row.update({
        "match_id": mid, "fecha": date, "home": home, "away": away,
        "equipos": f"{home} vs {away}", "pick": OUTCOMES[pick],
        "tu_prob": round(float(model_probs[pick]), 4),
        "odds_source": source, "open_at": datetime.now(timezone.utc).isoformat()})
    if odds_1x2 is not None:
        row.update({"cuota_apertura": round(float(odds_1x2[pick]), 3),
                    "open_1": odds_1x2[0], "open_X": odds_1x2[1],
                    "open_2": odds_1x2[2]})
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    _save(df)
    print(f"  ✅ apertura {mid}: pick={row['pick']} tu_prob={row['tu_prob']} "
          f"cuota_apertura={row['cuota_apertura']} ({source})")
    return row


def record_close(home, away, date, odds_1x2=None) -> dict:
    """Añade cuota de CIERRE y calcula CLV, sin tocar la apertura ya guardada."""
    df = _load()
    mid = _mid(home, away, date)
    sel = df["match_id"] == mid
    if not sel.any():
        raise ValueError(f"{mid} no tiene apertura registrada (usa --open antes).")
    i = df[sel].index[0]
    # las columnas de cierre se releen como float64 (NaN); pásalas a object para
    # poder escribir cuotas/strings sin conflicto de dtype (igual que log_friendlies).
    for col in ("cuota_cierre", "clv_resultante", "edge_vs_cierre",
                "close_1", "close_X", "close_2", "close_at"):
        if df[col].dtype != object:
            df[col] = df[col].astype(object)
    if odds_1x2 is None:
        fetched = fetch_odds_1x2(home, away, date)
        if fetched is None:
            print(f"  ⚠️ sin cuotas de cierre para {mid}; pendiente")
            return df.loc[i].to_dict()
        odds_1x2 = fetched[0]
    open_1x2 = [df.at[i, c] for c in ("open_1", "open_X", "open_2")]
    if any(pd.isna(x) for x in open_1x2):
        print(f"  ⚠️ {mid} no tiene cuota de apertura: no se puede calcular CLV "
              "de precio (la apertura no estaba disponible al predecir).")
        for c, v in zip(("close_1", "close_X", "close_2"), odds_1x2):
            df.at[i, c] = v
        df.at[i, "cuota_cierre"] = round(float(odds_1x2[OUTCOMES.index(
            df.at[i, "pick"])]), 3)
        df.at[i, "close_at"] = datetime.now(timezone.utc).isoformat()
        _save(df)
        return df.loc[i].to_dict()

    # model_probs reconstruidas del pick+tu_prob no bastan; usamos la apertura
    # des-marginada como proxy de tu_prob por resultado solo para elegir pick ya
    # fijado. El CLV se calcula para el PICK guardado.
    pick = df.at[i, "pick"]
    pidx = OUTCOMES.index(pick)
    model_probs = [0.0, 0.0, 0.0]
    model_probs[pidx] = float(df.at[i, "tu_prob"])
    # rellena las otras dos con la apertura des-marginada (no afecta el pick)
    open_fair = demargin([float(x) for x in open_1x2], method="shin")
    for k in range(3):
        if k != pidx:
            model_probs[k] = float(open_fair[k])
    res = compute_clv(model_probs, [float(x) for x in open_1x2], odds_1x2)
    df.at[i, "cuota_cierre"] = res["close_odds"]
    df.at[i, "clv_resultante"] = res["clv_price"]
    df.at[i, "edge_vs_cierre"] = res["edge_vs_close"]
    for c, v in zip(("close_1", "close_X", "close_2"), odds_1x2):
        df.at[i, c] = v
    df.at[i, "close_at"] = datetime.now(timezone.utc).isoformat()
    _save(df)
    print(f"  ✅ cierre {mid}: cuota_cierre={res['close_odds']} "
          f"CLV={res['clv_price']:+.4f} edge_vs_cierre={res['edge_vs_close']:+.4f}")
    return df.loc[i].to_dict()


def write_report() -> None:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    df = _load()
    done = df[df["clv_resultante"].notna()] if not df.empty else df
    s = summarize_clv(done["clv_resultante"].tolist()) if not done.empty \
        else {"n": 0, "clv_mean": None, "beat_close_rate": None}
    md = [
        "# B. CLV tracking — closing line value\n",
        "Registra, por partido, tu probabilidad y las cuotas de **apertura** y "
        "**cierre** del mercado para calcular el CLV — el mejor proxy de *edge* "
        "real (mejor que la accuracy: te dice si llegaste a la línea antes que el "
        "mercado).\n",
        "## Estado\n",
        f"- Partidos en `results/clv_tracking.csv`: **{len(df)}**.",
        f"- Con CLV calculado (apertura+cierre): **{s['n']}**.",
    ]
    if s["n"]:
        md += [f"- **CLV medio**: {s['clv_mean']:+.4f} "
               f"(>0 sostenido = edge real).",
               f"- **% de picks que batieron el cierre**: {s['beat_close_rate']:.0%}."]
    else:
        md += ["- Aún sin partidos cerrados: el CLV se valida con el tiempo, "
               "partido a partido, sobre amistosos/Mundial reales."]
    md += [
        "\n## Fuente de cuotas y LIMITACIONES (honesto)\n",
        "- **Fuente**: The Odds API (the-odds-api.com), tier gratuito, vía "
        "`ODDS_API_KEY` en `.env`. **Hoy NO hay clave configurada**, así que la "
        "captura automática de cuotas está INACTIVA: las funciones de red "
        "devuelven `None` y el CSV solo se llena con cuotas pasadas a mano. No se "
        "inventan cuotas.",
        "- **Cobertura**: el tier gratuito (~500 req/mes) cubre selecciones solo "
        "cuando hay torneo/amistoso listado por las casas EU/UK; muchos "
        "amistosos menores **no** aparecen.",
        "- **Mercado**: solo **1X2 (h2h)**. No incluye over/under de goles ni "
        "hándicaps — el CLV aquí es exclusivamente del resultado.",
        "- **Latencia/'apertura'**: el tier gratuito devuelve la cuota VIGENTE, "
        "no un histórico de aperturas. La 'apertura' que se guarda es la cuota al "
        "MOMENTO de predecir (early line) y el 'cierre' la del momento previo al "
        "partido: aproxima open/close con dos snapshots, no con el tick real de "
        "apertura de cada casa.",
        "- **Des-margining**: las cuotas se des-marginan (Shin) y se promedian en "
        "log entre casas (`core/data/odds_tools.py`) antes de comparar con tu "
        "probabilidad, para no confundir margen de la casa con edge.",
        "\n## Alternativas si se quiere cobertura real\n",
        "- Registrar una `ODDS_API_KEY` gratuita y correr `--open`/`--close` en "
        "los momentos correctos (idealmente automatizado cerca del kickoff).",
        "- O cargar cuotas a mano con `--odds 1 X 2` desde cualquier casa "
        "accesible (Pinnacle es el estándar de oro para CLV por su bajo margen).",
    ]
    REPORT.write_text("\n".join(md), encoding="utf-8")
    print(f"Reporte: {REPORT}")


# ----------------------------------- CLI ------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="CLV tracking (B)")
    ap.add_argument("--open", nargs=3, metavar=("HOME", "AWAY", "DATE"))
    ap.add_argument("--close", nargs=3, metavar=("HOME", "AWAY", "DATE"))
    ap.add_argument("--odds", nargs=3, type=float, metavar=("O1", "OX", "O2"),
                    help="cuotas decimales 1 X 2 (a mano)")
    ap.add_argument("--probs", nargs=3, type=float, metavar=("P1", "PX", "P2"),
                    help="tu probabilidad 1 X 2 (si no, se usa predict_final)")
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()

    if args.open:
        h, a, d = args.open
        probs = args.probs
        if probs is None:
            from ensemble.predict import predict_final
            p = predict_final(h, a, d)
            probs = [p.prob_home, p.prob_draw, p.prob_away]
        record_open(h, a, d, probs, odds_1x2=args.odds)
    if args.close:
        record_close(*args.close, odds_1x2=args.odds)
    write_report()


if __name__ == "__main__":
    main()
