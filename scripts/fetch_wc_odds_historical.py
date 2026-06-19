"""Intenta obtener odds 1X2 históricas de los Mundiales 2010-2022.

REALIDAD (verificada): OddsPortal NO sirve sus odds en el HTML — las carga por
XHR con codificación ofuscada y JS. `requests` recibe un cascarón vacío; haría
falta selenium/Playwright + reverse-engineering de su feed (ver el proyecto
OddsHarvester). Además NO existe un CSV público descargable de odds de partidos
de Mundial (sí de clubes: github.com/xgabora/Club-Football-Match-Data-2000-2025,
usado por validate_blend_alpha.py --source club).

Por eso este script:
  1) hace un INTENTO honesto con requests y reporta si no pudo extraer odds;
  2) acepta --input CSV para ingerir odds que consigas por otra vía;
  3) escribe files/f0_raw/wc_historical_odds.csv (vacío/plantilla si no hubo datos)
     con columnas: date,home_team,away_team,odds_home,odds_draw,odds_away,tournament_year

Uso:
  python scripts/fetch_wc_odds_historical.py            # intento + diagnóstico
  python scripts/fetch_wc_odds_historical.py --input mis_odds.csv
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import requests

OUT = ROOT / "files/f0_raw/wc_historical_odds.csv"
COLS = ["date", "home_team", "away_team", "odds_home", "odds_draw",
        "odds_away", "tournament_year"]
URLS = {
    2022: "https://www.oddsportal.com/soccer/world/world-cup-2022/results/",
    2018: "https://www.oddsportal.com/soccer/world/world-cup-2018/results/",
    2014: "https://www.oddsportal.com/soccer/world/world-cup-2014/results/",
    2010: "https://www.oddsportal.com/soccer/world/world-cup-2010/results/",
}
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
           "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"}


def try_oddsportal() -> pd.DataFrame:
    rows = []
    for year, url in URLS.items():
        try:
            r = requests.get(url, headers=HEADERS, timeout=20)
            html = r.text
            has_odds = any(k in html.lower() for k in
                           ('data-odd', '"odd"', 'odds_home'))
            n_events = html.lower().count('eventrow')
            print(f"  {year}: HTTP {r.status_code}, {len(html)} bytes, "
                  f"odds_en_html={has_odds}, filas_partido≈{n_events}")
        except Exception as e:
            print(f"  {year}: error {e}")
    print("\n→ Confirmado: OddsPortal no expone odds en el HTML (JS/XHR "
          "ofuscado). requests no puede extraerlas. Usa --input o un scraper "
          "dedicado (OddsHarvester + selenium).")
    return pd.DataFrame(columns=COLS)


def from_input(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    ren = {"home": "home_team", "away": "away_team", "year": "tournament_year",
           "season": "tournament_year"}
    df = df.rename(columns={k: v for k, v in ren.items() if k in df.columns})
    missing = set(COLS[:-1]) - set(df.columns)
    if missing:
        raise SystemExit(f"Al CSV de entrada le faltan columnas: {missing}")
    if "tournament_year" not in df.columns:
        df["tournament_year"] = ""
    for c in ("odds_home", "odds_draw", "odds_away"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["odds_home", "odds_draw", "odds_away"])[COLS]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", help="CSV con odds conseguidas por otra vía")
    args = ap.parse_args()

    if args.input:
        df = from_input(args.input)
        print(f"Ingeridas {len(df)} filas desde {args.input}")
    else:
        print("Intentando OddsPortal (diagnóstico honesto)...")
        df = try_oddsportal()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"\nEscrito {OUT} ({len(df)} filas).")
    if len(df) == 0:
        print("⚠ Sin odds de Mundial. validate_blend_alpha.py --source wc no "
              "tendrá datos; usa --source club para una validación medible.")


if __name__ == "__main__":
    main()
