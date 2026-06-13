"""Actualiza files/f0_raw/teams_2026.csv con datos reales.

- ELO: descargado en vivo de eloratings.net (World.tsv).
- Valor de mercado: Transfermarkt (snapshot junio 2026, tabla inline).
- xG a favor / en contra: derivados del ELO real con la misma forma
  funcional del generador de entrenamiento (proxy hasta tener datos de
  un proveedor: Opta/StatsBomb/FBref).
- Forma, caps, experiencia mundialista, lesiones, localía y distancias:
  tabla inline editable (estimaciones).

Uso:  python scripts/update_team_data.py
"""

import io
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

ELO_URL = "https://www.eloratings.net/World.tsv"

# Código eloratings.net de cada selección de la polla
ELO_CODES = {
    "México": "MX", "Sudáfrica": "ZA", "Corea del Sur": "KR",
    "Rep. Checa": "CZ", "Canadá": "CA", "Bosnia y Her.": "BA",
    "Catar": "QA", "Suiza": "CH", "Brasil": "BR", "Marruecos": "MA",
    "Haití": "HT", "Escocia": "SQ", "EEUU": "US", "Paraguay": "PY",
    "Australia": "AU", "Turquía": "TR", "Alemania": "DE", "Curazao": "CW",
    "Costa de Marfil": "CI", "Ecuador": "EC", "Países Bajos": "NL",
    "Japón": "JP", "Suecia": "SE", "Túnez": "TN", "Bélgica": "BE",
    "Egipto": "EG", "Irán": "IR", "Nueva Zelanda": "NZ", "España": "ES",
    "Cabo Verde": "CV", "Arabia S.": "SA", "Uruguay": "UY",
    "Francia": "FR", "Senegal": "SN", "Irak": "IQ", "Noruega": "NO",
    "Argentina": "AR", "Argelia": "DZ", "Austria": "AT", "Jordania": "JO",
    "Portugal": "PT", "RD Congo": "CD", "Uzbekistán": "UZ",
    "Colombia": "CO", "Inglaterra": "EN", "Croacia": "HR", "Ghana": "GH",
    "Panamá": "PA",
}

# Transfermarkt, valor total del plantel en M EUR (snapshot 2026-06-12)
MARKET_VALUES = {
    "España": 1220, "Francia": 1520, "Inglaterra": 1360, "Portugal": 1010,
    "Alemania": 947, "Brasil": 928, "Argentina": 807.5,
    "Países Bajos": 754.2, "Noruega": 589.9, "Bélgica": 547.5,
    "Costa de Marfil": 522.1, "Senegal": 478.1, "Turquía": 473.7,
    "Marruecos": 447.7, "Suecia": 406.1, "Croacia": 387.3, "EEUU": 385.7,
    "Ecuador": 368.7, "Uruguay": 359.3, "Suiza": 332.5, "Colombia": 302.4,
    "Japón": 270.9, "Argelia": 256.9, "Austria": 245.2, "Ghana": 234.6,
    "Canadá": 198.7, "México": 191.9, "Rep. Checa": 188.2,
    "Escocia": 170.3, "Paraguay": 153.7, "Bosnia y Her.": 146.4,
    "RD Congo": 143.9, "Corea del Sur": 139.1, "Egipto": 116.5,
    "Uzbekistán": 85.3, "Australia": 77.5, "Túnez": 70.0, "Haití": 55.9,
    "Cabo Verde": 54.5, "Sudáfrica": 49.3, "Arabia S.": 40.7,
    "Panamá": 34.6, "Nueva Zelanda": 34.4, "Irán": 32.1, "Curazao": 25.8,
    "Irak": 21.2, "Jordania": 20.3, "Catar": 19.9,
}

# Estimaciones editables: forma últimos 5 (% pts), caps promedio,
# jugadores con mundial previo, índice de bajas, localía, distancia (km)
EXTRAS = {  # team: (form, caps, wc_exp, injury, is_host, dist_km)
    "México": (0.60, 40, 14, 0.03, 1, 400),
    "Sudáfrica": (0.60, 30, 5, 0.03, 0, 14500),
    "Corea del Sur": (0.60, 38, 12, 0.04, 0, 10500),
    "Rep. Checa": (0.53, 32, 2, 0.03, 0, 8700),
    "Canadá": (0.60, 35, 10, 0.03, 1, 500),
    "Bosnia y Her.": (0.60, 33, 3, 0.04, 0, 9200),
    "Catar": (0.47, 42, 8, 0.03, 0, 12100),
    "Suiza": (0.67, 40, 15, 0.03, 0, 9200),
    "Brasil": (0.73, 30, 12, 0.04, 0, 7600),
    "Marruecos": (0.80, 35, 14, 0.03, 0, 7700),
    "Haití": (0.53, 28, 0, 0.04, 0, 2800),
    "Escocia": (0.67, 34, 2, 0.03, 0, 6900),
    "EEUU": (0.53, 34, 13, 0.03, 1, 300),
    "Paraguay": (0.60, 33, 6, 0.03, 0, 7400),
    "Australia": (0.60, 37, 11, 0.03, 0, 13500),
    "Turquía": (0.67, 30, 4, 0.03, 0, 9300),
    "Alemania": (0.67, 32, 10, 0.03, 0, 7900),
    "Curazao": (0.60, 28, 0, 0.03, 0, 3200),
    "Costa de Marfil": (0.67, 29, 5, 0.04, 0, 8900),
    "Ecuador": (0.67, 32, 9, 0.03, 0, 4500),
    "Países Bajos": (0.73, 33, 11, 0.02, 0, 7400),
    "Japón": (0.73, 38, 12, 0.02, 0, 10300),
    "Suecia": (0.60, 33, 5, 0.03, 0, 7800),
    "Túnez": (0.60, 35, 9, 0.03, 0, 9400),
    "Bélgica": (0.67, 38, 13, 0.03, 0, 7500),
    "Egipto": (0.67, 33, 8, 0.04, 0, 10400),
    "Irán": (0.60, 41, 13, 0.03, 0, 11400),
    "Nueva Zelanda": (0.60, 33, 4, 0.02, 0, 11300),
    "España": (0.87, 32, 11, 0.02, 0, 7800),
    "Cabo Verde": (0.60, 30, 0, 0.03, 0, 6500),
    "Arabia S.": (0.53, 39, 10, 0.03, 0, 11800),
    "Uruguay": (0.60, 35, 11, 0.03, 0, 8800),
    "Francia": (0.73, 33, 14, 0.03, 0, 7600),
    "Senegal": (0.73, 36, 11, 0.03, 0, 7900),
    "Irak": (0.53, 35, 2, 0.03, 0, 11500),
    "Noruega": (0.80, 28, 0, 0.02, 0, 7500),
    "Argentina": (0.80, 40, 16, 0.03, 0, 8500),
    "Argelia": (0.67, 33, 4, 0.03, 0, 8800),
    "Austria": (0.67, 36, 7, 0.02, 0, 8700),
    "Jordania": (0.53, 36, 0, 0.03, 0, 10900),
    "Portugal": (0.73, 34, 13, 0.03, 0, 6700),
    "RD Congo": (0.60, 28, 0, 0.04, 0, 11000),
    "Uzbekistán": (0.60, 34, 0, 0.03, 0, 11200),
    "Colombia": (0.67, 36, 10, 0.03, 0, 4300),
    "Inglaterra": (0.80, 30, 12, 0.02, 0, 7000),
    "Croacia": (0.67, 39, 15, 0.03, 0, 8600),
    "Ghana": (0.60, 30, 8, 0.04, 0, 9800),
    "Panamá": (0.60, 40, 7, 0.02, 0, 3500),
}

GROUPS = {
    "A": ["México", "Sudáfrica", "Corea del Sur", "Rep. Checa"],
    "B": ["Canadá", "Bosnia y Her.", "Catar", "Suiza"],
    "C": ["Brasil", "Marruecos", "Haití", "Escocia"],
    "D": ["EEUU", "Paraguay", "Australia", "Turquía"],
    "E": ["Alemania", "Curazao", "Costa de Marfil", "Ecuador"],
    "F": ["Países Bajos", "Japón", "Suecia", "Túnez"],
    "G": ["Bélgica", "Egipto", "Irán", "Nueva Zelanda"],
    "H": ["España", "Cabo Verde", "Arabia S.", "Uruguay"],
    "I": ["Francia", "Senegal", "Irak", "Noruega"],
    "J": ["Argentina", "Argelia", "Austria", "Jordania"],
    "K": ["Portugal", "RD Congo", "Uzbekistán", "Colombia"],
    "L": ["Inglaterra", "Croacia", "Ghana", "Panamá"],
}

CONFED = {
    "México": "CONCACAF", "Sudáfrica": "CAF", "Corea del Sur": "AFC",
    "Rep. Checa": "UEFA", "Canadá": "CONCACAF", "Bosnia y Her.": "UEFA",
    "Catar": "AFC", "Suiza": "UEFA", "Brasil": "CONMEBOL",
    "Marruecos": "CAF", "Haití": "CONCACAF", "Escocia": "UEFA",
    "EEUU": "CONCACAF", "Paraguay": "CONMEBOL", "Australia": "AFC",
    "Turquía": "UEFA", "Alemania": "UEFA", "Curazao": "CONCACAF",
    "Costa de Marfil": "CAF", "Ecuador": "CONMEBOL",
    "Países Bajos": "UEFA", "Japón": "AFC", "Suecia": "UEFA",
    "Túnez": "CAF", "Bélgica": "UEFA", "Egipto": "CAF", "Irán": "AFC",
    "Nueva Zelanda": "OFC", "España": "UEFA", "Cabo Verde": "CAF",
    "Arabia S.": "AFC", "Uruguay": "CONMEBOL", "Francia": "UEFA",
    "Senegal": "CAF", "Irak": "AFC", "Noruega": "UEFA",
    "Argentina": "CONMEBOL", "Argelia": "CAF", "Austria": "UEFA",
    "Jordania": "AFC", "Portugal": "UEFA", "RD Congo": "CAF",
    "Uzbekistán": "AFC", "Colombia": "CONMEBOL", "Inglaterra": "UEFA",
    "Croacia": "UEFA", "Ghana": "CAF", "Panamá": "CONCACAF",
}


def fetch_elos() -> dict[str, int]:
    with urllib.request.urlopen(ELO_URL, timeout=30) as r:
        raw = r.read().decode("utf-8")
    df = pd.read_csv(io.StringIO(raw), sep="\t", header=None,
                     usecols=[2, 3], names=["code", "elo"])
    by_code = df.drop_duplicates("code").set_index("code")["elo"].to_dict()
    missing = {t: c for t, c in ELO_CODES.items() if c not in by_code}
    if missing:
        raise SystemExit(f"Códigos ELO no encontrados: {missing}")
    return {t: int(by_code[c]) for t, c in ELO_CODES.items()}


def xg_from_elo(elo: float) -> tuple[float, float]:
    """Proxy de xG por partido derivado del ELO (misma forma funcional
    del generador de entrenamiento)."""
    z = (elo - 1500) / 170.0
    xg_for = float(np.clip(1.25 * np.exp(0.25 * z), 0.5, 3.2))
    xg_against = float(np.clip(1.25 * np.exp(-0.25 * z), 0.5, 3.2))
    return round(xg_for, 2), round(xg_against, 2)


def main() -> None:
    print(f"Descargando ELO actuales de {ELO_URL} ...")
    elos = fetch_elos()
    rows = []
    for g, names in GROUPS.items():
        for t in names:
            form, caps, wc_exp, injury, host, dist = EXTRAS[t]
            xf, xa = xg_from_elo(elos[t])
            rows.append({
                "team": t, "group": g, "confed": CONFED[t],
                "elo": elos[t], "xg_for_last10": xf, "xg_against_last10": xa,
                "form_last5_points_pct": form,
                "market_value_meur": MARKET_VALUES[t],
                "avg_caps": caps, "players_with_wc_experience": wc_exp,
                "injury_impact_index": injury, "is_host": host,
                "distance_avg_km": dist,
            })
    out = ROOT / "files/f0_raw/teams_2026.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"OK: {out} actualizado con ELO reales y valores Transfermarkt.")
    top = sorted(elos.items(), key=lambda kv: -kv[1])[:10]
    print("Top 10 ELO:", ", ".join(f"{t} {e}" for t, e in top))


if __name__ == "__main__":
    main()
