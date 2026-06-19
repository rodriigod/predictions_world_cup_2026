"""Scraper de odds 1X2 de Mundial desde OddsPortal con Selenium (OPCIONAL).

AVISO HONESTO: OddsPortal carga las odds por XHR con valores ofuscados y tiene
anti-bot. Incluso con Selenium suele requerir esperar el render, scrollear y a
veces resolver protecciones; el éxito NO está garantizado. Este entorno no trae
Selenium ni navegador instalado — corre este script en TU máquina:

    pip install selenium
    # y un navegador + driver (Chrome/Chromedriver o Firefox/geckodriver)

Si falla, genera una plantilla manual en files/f0_raw/wc_odds_oddsportal.csv
para que cargues a mano (date, home_team, away_team, odds_home, odds_draw,
odds_away, tournament_year).

Uso:
    python scripts/scrape_oddsportal_wc.py            # intenta selenium
    python scripts/scrape_oddsportal_wc.py --template # solo plantilla manual
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

OUT = ROOT / "files/f0_raw/wc_odds_oddsportal.csv"
COLS = ["date", "home_team", "away_team", "odds_home", "odds_draw",
        "odds_away", "tournament_year"]
URLS = {
    2022: "https://www.oddsportal.com/soccer/world/world-cup-2022/results/",
    2018: "https://www.oddsportal.com/soccer/world/world-cup-2018/results/",
    2014: "https://www.oddsportal.com/soccer/world/world-cup-2014/results/",
    2010: "https://www.oddsportal.com/soccer/world/world-cup-2010/results/",
}
DELAY = 5.0


def write_template() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=COLS).to_csv(OUT, index=False)
    print(f"Plantilla vacía escrita en {OUT}. Carga las filas a mano "
          "(odds decimales) y úsalas con validate_blend_alpha.py --source wc.")


def scrape() -> pd.DataFrame:
    try:
        from selenium import webdriver
        from selenium.webdriver.common.by import By
        from selenium.webdriver.chrome.options import Options
    except ImportError:
        print("✗ Selenium no está instalado (pip install selenium). "
              "Genero plantilla manual."); return pd.DataFrame(columns=COLS)

    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36")
    try:
        driver = webdriver.Chrome(options=opts)
    except Exception as e:
        print(f"✗ No pude iniciar Chrome/Chromedriver: {e}\n"
              "  Instala el navegador y el driver. Genero plantilla manual.")
        return pd.DataFrame(columns=COLS)

    rows = []
    try:
        for year, url in URLS.items():
            print(f"  {year}: {url}")
            driver.get(url)
            time.sleep(DELAY)              # respetar render + rate limit
            # OddsPortal usa filas con clase 'eventRow'; las odds están en
            # <p> dentro de cada fila. Selectores defensivos (cambian seguido).
            evs = driver.find_elements(By.CSS_SELECTOR, "div.eventRow")
            print(f"     filas detectadas: {len(evs)}")
            for ev in evs:
                try:
                    teams = ev.find_elements(By.CSS_SELECTOR, "a p")
                    odds = ev.find_elements(By.CSS_SELECTOR, "p.height-content")
                    if len(teams) >= 2 and len(odds) >= 3:
                        rows.append({
                            "date": "", "home_team": teams[0].text,
                            "away_team": teams[1].text,
                            "odds_home": _f(odds[0].text), "odds_draw": _f(odds[1].text),
                            "odds_away": _f(odds[2].text), "tournament_year": year})
                except Exception:
                    continue
    finally:
        driver.quit()
    df = pd.DataFrame(rows, columns=COLS).dropna(
        subset=["odds_home", "odds_draw", "odds_away"])
    return df


def _f(t):
    try:
        return float(str(t).strip())
    except ValueError:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", action="store_true")
    args = ap.parse_args()
    if args.template:
        write_template(); return
    df = scrape()
    if len(df) == 0:
        print("✗ No se extrajeron odds (anti-bot/ofuscación o sin selenium).")
        write_template(); return
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"✅ {len(df)} filas -> {OUT}")


if __name__ == "__main__":
    main()
