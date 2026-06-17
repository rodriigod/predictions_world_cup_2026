"""Descarga/prepara las odds 1X2 del mercado para los partidos del Mundial
2026 que quedan por jugar y las deja en files/f0_raw/odds_2026.csv con:

    date, home_team, away_team, odds_home, odds_draw, odds_away

Tres modos:
  1) API  (the-odds-api):  --api-key XXXX  (o variable de entorno ODDS_API_KEY)
        Baja odds h2h decimales en vivo y las mapea a los nombres del fixture.
  2) MANUAL: --input mis_odds.csv
        Limpia/valida un CSV que pegaste a mano de cualquier casa de apuestas
        (acepta nombres de equipo del fixture o en inglés del dataset).
  3) TEMPLATE (default si no hay key ni input): escribe el CSV con los partidos
        que faltan y odds vacías para que las rellenes a mano.

Las odds son DECIMALES (ej. 2.10). El blend en el simulador las pasa a
probabilidades sin margen.

Uso:
    export ODDS_API_KEY=...   &&  python scripts/fetch_odds_2026.py
    python scripts/fetch_odds_2026.py --input pegado.csv
    python scripts/fetch_odds_2026.py --template
"""
import argparse
import json
import os
import sys
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.data.historical import NAME_MAP   # español(polla) -> inglés(dataset)

OUT = ROOT / "files/f0_raw/odds_2026.csv"
FIXTURES = ROOT / "files/f0_raw/fixtures_2026.csv"
COLS = ["date", "home_team", "away_team", "odds_home", "odds_draw", "odds_away"]
SPORT = "soccer_fifa_world_cup"
API = "https://api.the-odds-api.com/v4/sports/{sport}/odds"

# inglés(dataset/API) -> español(polla). Incluye alias comunes de the-odds-api.
EN2ES = {v: k for k, v in NAME_MAP.items()}
EN2ES.update({
    "United States": "EEUU", "USA": "EEUU", "South Korea": "Corea del Sur",
    "Korea Republic": "Corea del Sur", "IR Iran": "Irán", "Iran": "Irán",
    "Czechia": "Rep. Checa", "Czech Republic": "Rep. Checa",
    "Cape Verde Islands": "Cabo Verde", "Ivory Coast": "Costa de Marfil",
    "Côte d'Ivoire": "Costa de Marfil", "Curacao": "Curazao",
    "Bosnia and Herzegovina": "Bosnia y Her.", "DR Congo": "RD Congo",
    "Saudi Arabia": "Arabia S.", "New Zealand": "Nueva Zelanda",
})


def remaining_fixtures(from_date: str) -> pd.DataFrame:
    fx = pd.read_csv(FIXTURES)
    fx = fx[fx["date"] >= from_date].copy()
    return fx[["date", "team_a", "team_b"]].rename(
        columns={"team_a": "home_team", "team_b": "away_team"})


def _to_es(name: str) -> str | None:
    if name in EN2ES:
        return EN2ES[name]
    # ya podría venir en español del fixture
    if name in NAME_MAP:
        return name
    return None


# --------------------------------- modos -------------------------------------
def from_api(api_key: str, regions: str, from_date: str) -> pd.DataFrame:
    url = API.format(sport=SPORT) + "?" + urllib.parse.urlencode({
        "apiKey": api_key, "regions": regions, "markets": "h2h",
        "oddsFormat": "decimal", "dateFormat": "iso"})
    print(f"[API] GET {SPORT} (regions={regions}) ...")
    with urllib.request.urlopen(url, timeout=30) as r:
        data = json.loads(r.read().decode())
    rows, unmapped = [], set()
    for ev in data:
        h_es, a_es = _to_es(ev.get("home_team", "")), _to_es(ev.get("away_team", ""))
        if not h_es or not a_es:
            unmapped.add((ev.get("home_team"), ev.get("away_team")))
            continue
        # mejores (máximas) odds entre casas, por outcome
        best = {"home": 0.0, "draw": 0.0, "away": 0.0}
        for bk in ev.get("bookmakers", []):
            for mk in bk.get("markets", []):
                if mk.get("key") != "h2h":
                    continue
                for o in mk.get("outcomes", []):
                    nm, price = o.get("name"), float(o.get("price", 0))
                    if nm == ev.get("home_team"):
                        best["home"] = max(best["home"], price)
                    elif nm == ev.get("away_team"):
                        best["away"] = max(best["away"], price)
                    elif nm and nm.lower() == "draw":
                        best["draw"] = max(best["draw"], price)
        if min(best.values()) <= 0:
            continue
        rows.append({"date": ev.get("commence_time", "")[:10],
                     "home_team": h_es, "away_team": a_es,
                     "odds_home": best["home"], "odds_draw": best["draw"],
                     "odds_away": best["away"]})
    if unmapped:
        print(f"[API] {len(unmapped)} partidos sin mapear (nombres): "
              f"{sorted(unmapped)[:5]}{' ...' if len(unmapped) > 5 else ''}")
    print(f"[API] {len(rows)} partidos con odds mapeados al fixture.")
    return pd.DataFrame(rows, columns=COLS)


def from_input(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    ren = {"home": "home_team", "away": "away_team", "team_a": "home_team",
           "team_b": "away_team", "1": "odds_home", "X": "odds_draw",
           "2": "odds_away", "odd_home": "odds_home", "odd_draw": "odds_draw",
           "odd_away": "odds_away"}
    df = df.rename(columns={k: v for k, v in ren.items() if k in df.columns})
    missing = set(COLS[1:]) - set(df.columns)
    if missing:
        raise SystemExit(f"Al CSV de entrada le faltan columnas: {missing}")
    if "date" not in df.columns:
        df["date"] = ""
    # normalizar nombres a español del fixture cuando se pueda
    df["home_team"] = df["home_team"].map(lambda n: _to_es(n) or n)
    df["away_team"] = df["away_team"].map(lambda n: _to_es(n) or n)
    for c in ["odds_home", "odds_draw", "odds_away"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["odds_home", "odds_draw", "odds_away"])
    return df[COLS]


def write_template(from_date: str) -> pd.DataFrame:
    fx = remaining_fixtures(from_date)
    for c in ["odds_home", "odds_draw", "odds_away"]:
        fx[c] = ""
    return fx[COLS]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-key", default=os.environ.get("ODDS_API_KEY"))
    ap.add_argument("--regions", default="eu")
    ap.add_argument("--input", help="CSV manual a limpiar")
    ap.add_argument("--template", action="store_true",
                    help="escribe plantilla vacía para rellenar a mano")
    ap.add_argument("--from-date", default=str(date.today()),
                    help="solo partidos desde esta fecha (default: hoy)")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    if args.input:
        df = from_input(args.input)
        mode = f"manual ({args.input})"
    elif args.api_key and not args.template:
        df = from_api(args.api_key, args.regions, args.from_date)
        mode = "the-odds-api"
    else:
        df = write_template(args.from_date)
        mode = "plantilla (rellenar a mano)"

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    have = df[["odds_home"]].apply(pd.to_numeric, errors="coerce").notna().sum().item() \
        if len(df) else 0
    print(f"\n✅ {mode}: {len(df)} filas -> {args.out}  ({have} con odds)")
    if mode.startswith("plantilla"):
        print("   Rellena odds_home/odds_draw/odds_away (decimales) y luego:")
        print("   python scripts/run_groups_simulation.py --alpha 0.3")


if __name__ == "__main__":
    main()
