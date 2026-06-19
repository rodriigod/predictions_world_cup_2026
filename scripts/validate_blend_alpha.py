"""¿Cuánto peso darle al mercado en el blend? — medición objetiva del α.

No existen odds históricas gratis de Mundial (OddsPortal es JS ofuscado; las
APIs cobran lo histórico). Pero el blend es domain-agnóstico, así que se valida
el MÉTODO sobre 230k partidos de CLUBES que SÍ traen odds Bet365 + ELO
pre-partido (github.com/xgabora/Club-Football-Match-Data-2000-2025).

Protocolo leak-free:
  1) modelo base = logística multinomial sobre (ELO_local-ELO_visita), ajustada
     SOLO con partidos anteriores al corte (train), predicha en test (futuro);
  2) mercado = odds Bet365 de-margined (1/odd normalizado);
  3) blend log-lineal p ∝ p_modelo^α · p_mercado^(1-α), α de 0.0 a 1.0;
  4) RPS por α en el test -> α óptimo + gráfico results/alpha_validation.png.

α=1.0 = solo modelo, α=0.0 = solo mercado. Dice cuánto confiar en cada uno.

Uso:
  python scripts/validate_blend_alpha.py                 # --source club (medible)
  python scripts/validate_blend_alpha.py --source wc     # usa wc_historical_odds.csv
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

CLUB_CSV = ROOT / "files/cache/club_matches.csv"
WC_ODDS = ROOT / "files/f0_raw/wc_historical_odds.csv"
PNG = ROOT / "results/alpha_validation.png"
EPS = 1e-9


def rps(y_idx: np.ndarray, proba: np.ndarray) -> float:
    """RPS ordinal (clases 0,1,2 = H/D/A). Menor = mejor."""
    cp = np.cumsum(proba, axis=1)
    ct = np.cumsum(np.eye(3)[y_idx], axis=1)
    return float(np.mean(np.sum((cp - ct) ** 2, axis=1) / 2))


def demargin(oh, od, oa) -> np.ndarray:
    imp = np.column_stack([1 / oh, 1 / od, 1 / oa])
    return imp / imp.sum(axis=1, keepdims=True)


def blend(p_model, p_market, alpha) -> np.ndarray:
    b = (np.maximum(p_model, EPS) ** alpha) * (np.maximum(p_market, EPS) ** (1 - alpha))
    return b / b.sum(axis=1, keepdims=True)


def load_club() -> pd.DataFrame:
    if not CLUB_CSV.exists():
        raise SystemExit(
            f"Falta {CLUB_CSV}. Bájalo:\n  curl -sL https://raw."
            "githubusercontent.com/xgabora/Club-Football-Match-Data-2000-2025/"
            "main/data/Matches.csv -o files/cache/club_matches.csv")
    df = pd.read_csv(CLUB_CSV, usecols=[
        "MatchDate", "HomeElo", "AwayElo", "FTResult",
        "OddHome", "OddDraw", "OddAway"], parse_dates=["MatchDate"])
    df = df.dropna()
    df = df[df["FTResult"].isin(["H", "D", "A"])]
    for c in ("OddHome", "OddDraw", "OddAway"):
        df = df[df[c] > 1.0]
    df["y"] = df["FTResult"].map({"H": 0, "D": 1, "A": 2})
    return df.sort_values("MatchDate").reset_index(drop=True)


def sweep_alpha(p_model, p_market, y_idx):
    alphas = np.round(np.arange(0.0, 1.0001, 0.1), 2)
    return alphas, [rps(y_idx, blend(p_model, p_market, a)) for a in alphas]


def run_club() -> None:
    df = load_club()
    n = len(df)
    split = int(n * 0.7)
    tr, te = df.iloc[:split], df.iloc[split:]
    print(f"[club] {n:,} partidos con ELO+odds | train {len(tr):,} "
          f"(≤{tr['MatchDate'].max().date()}) test {len(te):,} "
          f"(≥{te['MatchDate'].min().date()})")

    # modelo base: logística sobre la diferencia de ELO (leak-free)
    Xtr = tr[["HomeElo", "AwayElo"]].assign(diff=lambda d: d.HomeElo - d.AwayElo)[["diff"]]
    Xte = te[["HomeElo", "AwayElo"]].assign(diff=lambda d: d.HomeElo - d.AwayElo)[["diff"]]
    clf = LogisticRegression(max_iter=1000).fit(Xtr, tr["y"])
    p_model = clf.predict_proba(Xte)                      # columnas 0,1,2 = H,D,A
    p_market = demargin(te["OddHome"].values, te["OddDraw"].values,
                        te["OddAway"].values)
    y = te["y"].values

    rps_model = rps(y, p_model)
    rps_market = rps(y, p_market)
    alphas, scores = sweep_alpha(p_model, p_market, y)
    best_i = int(np.argmin(scores))
    best_alpha, best_rps = alphas[best_i], scores[best_i]

    print(f"\n  RPS solo-modelo (α=1.0):  {rps_model:.4f}")
    print(f"  RPS solo-mercado (α=0.0): {rps_market:.4f}")
    print(f"  {'α':>5} {'RPS':>8}")
    for a, s in zip(alphas, scores):
        mark = "  <- óptimo" if a == best_alpha else ""
        print(f"  {a:>5.1f} {s:>8.4f}{mark}")
    print(f"\n  ★ α óptimo = {best_alpha:.1f}  (RPS {best_rps:.4f})")
    gain = min(rps_model, rps_market) - best_rps
    print(f"  El blend {'MEJORA' if gain > 1e-4 else 'NO mejora'} al mejor "
          f"individual por {gain:+.4f} de RPS.")
    print(f"  Lectura: peso del MODELO ≈ {best_alpha:.0%}, "
          f"del MERCADO ≈ {1-best_alpha:.0%}.")

    _plot(alphas, scores, rps_model, rps_market, best_alpha,
          "Validación del blend (230k partidos de clubes, leak-free)")


def run_wc() -> None:
    if not WC_ODDS.exists() or len(pd.read_csv(WC_ODDS)) == 0:
        raise SystemExit(
            "No hay odds históricas de Mundial (wc_historical_odds.csv vacío). "
            "OddsPortal no es scrapeable y no hay CSV público. Usa --source club.")
    print("Modo WC: correría el modelo internacional walk-forward sobre estos "
          "partidos y barrería α. (Listo para cuando existan datos reales.)")
    # Implementación: análoga a run_club pero usando build_historical_dataset +
    # Poisson+DC sobre los partidos del CSV. Sin datos, no se ejecuta.


def _plot(alphas, scores, rps_model, rps_market, best_alpha, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(alphas, scores, "o-", label="blend")
    ax.axhline(rps_model, ls="--", c="tab:orange", label=f"solo modelo ({rps_model:.4f})")
    ax.axhline(rps_market, ls="--", c="tab:green", label=f"solo mercado ({rps_market:.4f})")
    ax.axvline(best_alpha, ls=":", c="red", label=f"α óptimo = {best_alpha:.1f}")
    ax.set_xlabel("α  (peso del modelo; 0=solo mercado, 1=solo modelo)")
    ax.set_ylabel("RPS (menor = mejor)")
    ax.set_title(title)
    ax.legend(); ax.grid(True, alpha=0.3)
    PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(PNG, dpi=150); plt.close(fig)
    print(f"  Gráfico: {PNG}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["club", "wc"], default="club")
    args = ap.parse_args()
    run_club() if args.source == "club" else run_wc()


if __name__ == "__main__":
    main()
