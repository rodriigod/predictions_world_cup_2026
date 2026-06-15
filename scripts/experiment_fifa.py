"""Test marginal HONESTO: ¿el ranking FIFA agrega señal sobre el ELO interno?

Para cada partido toma los puntos FIFA más recientes ANTES del partido (sin
leak), arma fifa_diff = log(pts_local) - log(pts_visita) y mide si sumarlo a
una logística 1X2 (que ya usa el ELO) mejora en una validación cronológica de
partidos NO-Mundial. Si no mejora, el ELO ya contiene esa información.

Uso: python scripts/experiment_fifa.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.historical import build_historical_dataset
from src.utils.metrics import ModelMetrics

CLASSES = ["1", "X", "2"]
IDX = {c: i for i, c in enumerate(CLASSES)}
# mapeo de nombres FIFA -> nombres del dataset de resultados (los grandes)
FIFA_NAME_FIX = {
    "Korea Republic": "South Korea", "USA": "United States",
    "IR Iran": "Iran", "Côte d'Ivoire": "Ivory Coast", "China PR": "China",
    "Czechia": "Czech Republic", "Cabo Verde": "Cape Verde",
    "Congo DR": "DR Congo", "Türkiye": "Turkey",
}


FIFA_CSV = ROOT / "files/f0_raw/fifa_ranking.csv"
FIFA_URL = ("https://raw.githubusercontent.com/Dato-Futbol/fifa-ranking/"
            "master/ranking_fifa_historical.csv")


def _fifa_diff(home, away, dates) -> np.ndarray:
    if not FIFA_CSV.exists():
        import urllib.request
        print(f"  descargando ranking FIFA de {FIFA_URL} ...")
        urllib.request.urlretrieve(FIFA_URL, FIFA_CSV)
    fifa = pd.read_csv(FIFA_CSV, parse_dates=["date"])
    fifa["team"] = fifa["team"].replace(FIFA_NAME_FIX)
    fifa = fifa[["team", "date", "total_points"]].sort_values("date")
    m = pd.DataFrame({"home": list(home), "away": list(away),
                      "date": pd.to_datetime(list(dates))}).reset_index()
    m = m.sort_values("date")

    def pts(col):
        j = pd.merge_asof(m, fifa.rename(columns={"team": col}),
                          on="date", by=col, direction="backward")
        return j.set_index("index")["total_points"]

    ph, pa = pts("home"), pts("away")
    diff = np.log(ph) - np.log(pa)
    return diff.reindex(range(len(m))).to_numpy()


def _metrics(model, X, y):
    proba = model.predict_proba(X)
    pos = {c: i for i, c in enumerate(model.classes_)}
    p = proba[:, [pos["1"], pos["X"], pos["2"]]]
    idx = np.array([IDX[v] for v in y])
    return (float((p.argmax(1) == idx).mean()),
            ModelMetrics.multiclass_logloss(list(y), p, CLASSES),
            ModelMetrics.rps(idx, p))


def main():
    print("Construyendo dataset y uniendo ranking FIFA (merge_asof, sin leak)...")
    data = build_historical_dataset(cutoff=None)
    Xm = data["X_match"].reset_index(drop=True)
    y = data["y_result"].reset_index(drop=True)
    md = data["match_dates"].to_numpy()
    fdiff = _fifa_diff(data["match_home"], data["match_away"], md)

    # filtros: no-Mundial, con dato FIFA, y partidos "modernos" (desde 2010)
    ok = (~data["match_is_wc"]) & np.isfinite(fdiff) & (md >= np.datetime64("2010-01-01"))
    Xm, y, md, fdiff = Xm[ok], y[ok].to_numpy(), md[ok], fdiff[ok]
    Xm = Xm.assign(fifa_diff=fdiff)
    print(f"  partidos usables: {len(Xm):,}")
    print(f"  corr(fifa_diff, elo_diff_scaled) = "
          f"{np.corrcoef(fdiff, Xm['elo_diff_scaled'])[0,1]:.3f}")

    # split cronológico: train < 2022, val >= 2022
    split = np.datetime64("2022-01-01")
    tr, va = md < split, md >= split
    base_feats = [c for c in Xm.columns if c != "fifa_diff"]
    print(f"  train: {tr.sum():,}  val: {va.sum():,}\n")

    def fit_eval(cols, label):
        model = Pipeline([("s", StandardScaler()),
                          ("lr", LogisticRegression(max_iter=2000, C=1.0))])
        model.fit(Xm[cols].iloc[tr], y[tr])
        acc, ll, rps = _metrics(model, Xm[cols].iloc[va], y[va])
        print(f"  {label:>28}: acc={acc:.3f}  log-loss={ll:.3f}  RPS={rps:.4f}")

    print("VALIDACIÓN (partidos no-Mundial desde 2022):")
    fit_eval(base_feats, "ELO + features actuales")
    fit_eval(base_feats + ["fifa_diff"], "+ ranking FIFA")
    fit_eval(["elo_diff_scaled", "elo_win_expectancy"], "solo ELO")
    fit_eval(["fifa_diff"], "solo ranking FIFA")


if __name__ == "__main__":
    main()
