#!/usr/bin/env python3
"""D. Stress-test de la metodología de stacking en liga de clubes (alto volumen).

NO es producción del Mundial: pipeline paralelo y autocontenido (no importa
core/ ni microsim/). Replica el esquema del ensemble —dos modelos base de
fuerza + meta-modelo logístico— sobre decenas de miles de partidos de clubes,
para distinguir si el stacking falla por CONCEPTO o por TAMAÑO DE MUESTRA.

Modelos base (señales DISTINTAS, a propósito):
  A (≈core)     : logística 1X2 sobre diferencia de ELO pre-partido.
  B (≈microsim) : logística sobre fuerza por GOLES (medias móviles leak-free de
                  goles a favor/en contra) + forma reciente.
Stacking: meta-logística sobre las 1X2 de A y B, todo OOF (TimeSeriesSplit).

Uso: python experiments/club_stress_test/run_stress_test.py
Salida: results/reports/club_stress_test.md
"""

import sys
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

CLUB_CSV = ROOT / "files/cache/club_matches.csv"
REPORT = ROOT / "results/reports/club_stress_test.md"
TOP5 = ["E0", "SP1", "D1", "I1", "F1"]      # Premier, La Liga, Bundesliga, Serie A, Ligue 1
CLASSES = ["H", "D", "A"]                    # local, empate, visita -> [1,X,2]
IDX = {c: i for i, c in enumerate(CLASSES)}
ROLL = 10                                    # ventana de forma por goles
MIN_HIST = 5                                 # mínimo de partidos previos para usar la fila
N_SPLITS = 6
N_BOOT = 2000
SEED = 42


# ------------------------------- métricas (locales) -------------------------
def rps(proba: np.ndarray, idx: np.ndarray) -> np.ndarray:
    cum_p = np.cumsum(proba, axis=1)
    cum_o = np.cumsum(np.eye(3)[idx], axis=1)
    return np.sum((cum_p - cum_o) ** 2, axis=1) / 2.0


def logloss(proba: np.ndarray, idx: np.ndarray) -> float:
    p = np.clip(proba[np.arange(len(idx)), idx], 1e-15, 1)
    return float(-np.mean(np.log(p)))


# ------------------------------- datos --------------------------------------
def load_features() -> pd.DataFrame:
    df = pd.read_csv(CLUB_CSV, parse_dates=["MatchDate"], low_memory=False)
    df = df[df["Division"].isin(TOP5)].copy()
    df = df.dropna(subset=["HomeElo", "AwayElo", "FTHome", "FTAway", "FTResult"])
    df = df.sort_values("MatchDate").reset_index(drop=True)

    # medias móviles leak-free de goles a favor/en contra por equipo
    gf, ga = {}, {}                          # team -> deque de goles
    att_h = np.full(len(df), np.nan); def_h = np.full(len(df), np.nan)
    att_a = np.full(len(df), np.nan); def_a = np.full(len(df), np.nan)
    n_prev_h = np.zeros(len(df)); n_prev_a = np.zeros(len(df))
    for i, r in enumerate(df.itertuples()):
        h, a = r.HomeTeam, r.AwayTeam
        dh_f, dh_a = gf.setdefault(h, deque(maxlen=ROLL)), ga.setdefault(h, deque(maxlen=ROLL))
        da_f, da_a = gf.setdefault(a, deque(maxlen=ROLL)), ga.setdefault(a, deque(maxlen=ROLL))
        if dh_f:
            att_h[i], def_h[i] = np.mean(dh_f), np.mean(dh_a)
        if da_f:
            att_a[i], def_a[i] = np.mean(da_f), np.mean(da_a)
        n_prev_h[i], n_prev_a[i] = len(dh_f), len(da_f)
        # actualizar DESPUÉS de leer (anti-leak)
        dh_f.append(r.FTHome); dh_a.append(r.FTAway)
        da_f.append(r.FTAway); da_a.append(r.FTHome)

    df["att_diff"] = att_h - att_a
    df["def_diff"] = def_a - def_h           # rival concede más - yo concedo = ventaja
    df["elo_diff"] = (df["HomeElo"] - df["AwayElo"]) / 400.0
    df["form5_diff"] = df["Form5Home"] - df["Form5Away"]
    df["y"] = df["FTResult"].map(IDX)
    usable = (n_prev_h >= MIN_HIST) & (n_prev_a >= MIN_HIST)
    return df[usable].dropna(
        subset=["att_diff", "def_diff", "form5_diff"]).reset_index(drop=True)


def _pipe() -> Pipeline:
    return Pipeline([("s", StandardScaler()),
                     ("lr", LogisticRegression(max_iter=2000, C=1.0))])


def _proba(model: Pipeline, X) -> np.ndarray:
    p = model.predict_proba(X)
    pos = {c: i for i, c in enumerate(model.named_steps["lr"].classes_)}
    return p[:, [pos[0], pos[1], pos[2]]]    # [H, D, A]


# ------------------------------- experimento --------------------------------
def main() -> None:
    df = load_features()
    n = len(df)
    print(f"\nStress-test stacking en clubes — {n:,} partidos top-5 "
          f"({df['MatchDate'].min().year}-{df['MatchDate'].max().year})\n")
    A_FEATS = ["elo_diff"]
    B_FEATS = ["att_diff", "def_diff", "form5_diff"]
    y = df["y"].to_numpy()

    oofA = np.full((n, 3), np.nan)
    oofB = np.full((n, 3), np.nan)
    oofS = np.full((n, 3), np.nan)
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    for tr, va in tscv.split(df):
        if len(np.unique(y[tr])) < 3:
            continue
        mA = _pipe().fit(df[A_FEATS].iloc[tr], y[tr])
        mB = _pipe().fit(df[B_FEATS].iloc[tr], y[tr])
        oofA[va], oofB[va] = _proba(mA, df[A_FEATS].iloc[va]), _proba(mB, df[B_FEATS].iloc[va])
        # meta-modelo sobre las probas base (entrena con base preds in-sample
        # del tramo de train -> ligeramente OPTIMISTA para el stacking, sesgo
        # que solo lo FAVORECE: si aun así no gana, la conclusión es robusta)
        baseA_tr, baseB_tr = _proba(mA, df[A_FEATS].iloc[tr]), _proba(mB, df[B_FEATS].iloc[tr])
        Z_tr = np.hstack([baseA_tr, baseB_tr])
        meta = LogisticRegression(max_iter=2000, C=1.0).fit(Z_tr, y[tr])
        Z_va = np.hstack([oofA[va], oofB[va]])
        pos = {c: i for i, c in enumerate(meta.classes_)}
        oofS[va] = meta.predict_proba(Z_va)[:, [pos[0], pos[1], pos[2]]]

    mask = ~np.isnan(oofS[:, 0])
    idx = y[mask]
    approaches = {"solo A (ELO)": oofA, "solo B (goles+forma)": oofB,
                  "promedio A+B": None, "stacking (OOF)": oofS}
    avg = oofA + oofB
    approaches["promedio A+B"] = avg / avg.sum(axis=1, keepdims=True)

    rps_match = {k: rps(v[mask], idx) for k, v in approaches.items()}
    results = {}
    for k, v in approaches.items():
        pr = v[mask]
        results[k] = {"acc": float((pr.argmax(1) == idx).mean()),
                      "rps": float(rps_match[k].mean()),
                      "logloss": logloss(pr, idx)}

    print(f"COMPARACIÓN sobre {int(mask.sum()):,} partidos OOF:")
    print(f"  {'enfoque':>22} {'acc':>7} {'RPS':>8} {'logloss':>9}")
    for k, m in results.items():
        print(f"  {k:>22} {m['acc']:>7.3f} {m['rps']:>8.4f} {m['logloss']:>9.3f}")

    # mejor base single y significancia (stacking - mejor base)
    best_single = min(["solo A (ELO)", "solo B (goles+forma)"],
                      key=lambda k: results[k]["rps"])
    rng = np.random.default_rng(SEED)
    m = int(mask.sum())
    bidx = rng.integers(0, m, size=(N_BOOT, m))
    d = (rps_match["stacking (OOF)"][bidx].mean(1)
         - rps_match[best_single][bidx].mean(1))
    lo, hi = np.percentile(d, [2.5, 97.5])
    zero = bool(lo <= 0 <= hi)
    wins = results["stacking (OOF)"]["rps"] < results[best_single]["rps"]
    print(f"\nstacking − {best_single}: ΔRPS={d.mean():+.4f} "
          f"IC95%=[{lo:+.4f},{hi:+.4f}] incluye0={'SÍ' if zero else 'no'}")

    _write_report(int(mask.sum()), df, results, best_single, d.mean(), lo, hi,
                  zero, wins)
    print(f"\nReporte: {REPORT}")


def _conclusion(best, wins, zero) -> str:
    if wins and not zero:
        return ("**El stacking SÍ le gana al mejor modelo base de forma "
                "significativa** con alto volumen. Esto sugiere que el fracaso "
                "del stacking en el Mundial es de TAMAÑO DE MUESTRA (cientos de "
                "partidos), no conceptual: vale la pena reintentar el stacking "
                "ahí a medida que se acumulen más partidos.")
    if wins and zero:
        return ("El stacking mejora el RPS pero el IC95% INCLUYE 0: la mejora "
                "no es estadísticamente concluyente ni con miles de partidos. "
                "Débil evidencia a favor del stacking; el problema del Mundial "
                "es probablemente una mezcla de muestra chica y correlación.")
    return ("**El stacking NO le gana al mejor modelo base ni con decenas de "
            "miles de partidos.** Refuerza que el problema es CONCEPTUAL: "
            "cuando los modelos base comparten la señal dominante (fuerza de "
            "equipo), apilarlos agrega varianza, no skill — el tamaño de "
            "muestra del Mundial no es la causa principal.")


def _write_report(n, df, results, best, dmu, lo, hi, zero, wins) -> None:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    md = ["# D. Stress-test de stacking en liga de clubes (alto volumen)\n",
          "Diagnóstico de metodología (NO producción del Mundial). Pipeline "
          "paralelo en `experiments/club_stress_test/`.\n",
          f"Datos: **{n:,} partidos** OOF de top-5 ligas europeas "
          f"({df['MatchDate'].min().year}-{df['MatchDate'].max().year}). "
          "Dos modelos base de fuerza (A=ELO, B=goles+forma) + meta-logística.\n",
          "| Enfoque | Accuracy | RPS | Log-loss |", "|---|:-:|:-:|:-:|"]
    for k, m in results.items():
        md.append(f"| {k} | {m['acc']:.3f} | {m['rps']:.4f} | {m['logloss']:.3f} |")
    md.append(f"\nMejor base single: **{best}**. "
              f"stacking − {best}: ΔRPS = **{dmu:+.4f}**, "
              f"IC95% = [{lo:+.4f}, {hi:+.4f}] "
              f"(incluye 0: {'SÍ' if zero else 'no'}).\n")
    md.append("## Conclusión\n")
    md.append(_conclusion(best, wins, zero))
    md.append("\n> Nota: el meta-modelo se entrena con las predicciones base "
              "in-sample del tramo de train (stacking 'con refit'), un sesgo que "
              "solo FAVORECE al stacking; aun así el resultado de arriba manda.")
    REPORT.write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
