"""D. Dataset de SECUENCIAS de forma por selección (separado de core/).

Recorre `international_results.csv` una sola vez y, para cada partido "usable"
(misma época que core/: desde 1995, con ambas selecciones con ≥N partidos
previos), emite:
  - home_seq, away_seq : (N, F) con los ÚLTIMOS N partidos de cada selección
    ANTES de este (anti-leakage), con features por partido:
       [win, draw, loss, gf/6, ga/6, home_flag, opp_elo_z, self_elo_z]
  - pi_feats : pi-ratings ataque/defensa + ΔElo (la "fuerza de equipo" actual que
    el experimento quiere comparar/complementar) — mismas constantes que core/.
  - label : 0=gana local, 1=empate, 2=gana visita (orden [1,X,2], como el RPS).
  - date : para el TimeSeriesSplit temporal.

NO toca core/. Replica su ELO/pi-rating online para que la comparación sea justa.
"""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.data.historical import (PI_CLIP, PI_HOME, PI_LR, PI_MU, TRAIN_FROM,
                                  load_results)

SEQ_FEATS = ["win", "draw", "loss", "gf", "ga", "home", "opp_elo_z", "self_elo_z"]
F = len(SEQ_FEATS)
PI_FEATS = ["home_att", "home_dfn", "away_att", "away_dfn", "elo_diff_z"]

HOME_ADV = 100.0


class _T:
    __slots__ = ("elo", "att", "dfn", "seq", "n")

    def __init__(self, N):
        self.elo = 1500.0
        self.att = 0.0
        self.dfn = 0.0
        self.seq: deque = deque(maxlen=N)
        self.n = 0


def build(N: int = 10) -> dict:
    """Construye el dataset de secuencias de longitud N. Devuelve arrays float32."""
    df = load_results()
    df = df[df["home_score"].notna()].reset_index(drop=True)
    hist: dict[str, _T] = {}
    train_from = np.datetime64(TRAIN_FROM)

    Hs, As, Pi, Y, D = [], [], [], [], []
    for r in df.itertuples():
        a = hist.setdefault(r.home_team, _T(N))
        b = hist.setdefault(r.away_team, _T(N))
        gh, ga = int(r.home_score), int(r.away_score)
        neutral = bool(r.neutral)

        usable = (np.datetime64(r.date) >= train_from
                  and len(a.seq) >= N and len(b.seq) >= N)
        if usable:
            Hs.append(np.array(a.seq, dtype=np.float32))
            As.append(np.array(b.seq, dtype=np.float32))
            Pi.append([a.att, a.dfn, b.att, b.dfn, (a.elo - b.elo) / 200.0])
            Y.append(0 if gh > ga else (2 if gh < ga else 1))
            D.append(r.date)

        # ---- features del partido para la secuencia de cada equipo ----
        home_flag = 0.5 if neutral else 1.0
        a_row = [1.0 if gh > ga else 0.0, 1.0 if gh == ga else 0.0,
                 1.0 if gh < ga else 0.0, min(gh, 6) / 6.0, min(ga, 6) / 6.0,
                 home_flag, (b.elo - 1500) / 200.0, (a.elo - 1500) / 200.0]
        b_row = [1.0 if ga > gh else 0.0, 1.0 if gh == ga else 0.0,
                 1.0 if ga < gh else 0.0, min(ga, 6) / 6.0, min(gh, 6) / 6.0,
                 0.0 if not neutral else 0.5, (a.elo - 1500) / 200.0,
                 (b.elo - 1500) / 200.0]
        a.seq.append(a_row)
        b.seq.append(b_row)

        # ---- update ELO + pi-ratings (idéntico a core/historical.py) ----
        adv = 0.0 if neutral else HOME_ADV
        we = 1.0 / (1.0 + 10 ** (-(a.elo + adv - b.elo) / 400.0))
        sc = 1.0 if gh > ga else (0.5 if gh == ga else 0.0)
        d = 40.0 * (sc - we)
        a.elo += d
        b.elo -= d
        pi_home = 0.0 if neutral else PI_HOME
        pred_a = min(np.exp(PI_MU + a.att - b.dfn + pi_home), 6.0)
        pred_b = min(np.exp(PI_MU + b.att - a.dfn), 6.0)
        err_a, err_b = gh - pred_a, ga - pred_b
        a.att = float(np.clip(a.att + PI_LR * err_a, -PI_CLIP, PI_CLIP))
        b.dfn = float(np.clip(b.dfn - PI_LR * err_a, -PI_CLIP, PI_CLIP))
        b.att = float(np.clip(b.att + PI_LR * err_b, -PI_CLIP, PI_CLIP))
        a.dfn = float(np.clip(a.dfn - PI_LR * err_b, -PI_CLIP, PI_CLIP))
        a.n += 1
        b.n += 1

    order = np.argsort(np.array(D, dtype="datetime64[D]"), kind="stable")
    return {
        "home_seq": np.stack(Hs)[order],
        "away_seq": np.stack(As)[order],
        "pi": np.asarray(Pi, dtype=np.float32)[order],
        "y": np.asarray(Y, dtype=np.int64)[order],
        "dates": np.array(D, dtype="datetime64[D]")[order],
        "N": N, "F": F, "pi_feats": PI_FEATS,
    }


if __name__ == "__main__":
    for N in (10, 20):
        d = build(N)
        print(f"N={N}: {len(d['y'])} partidos | home_seq {d['home_seq'].shape} | "
              f"pi {d['pi'].shape} | clases {np.bincount(d['y'])}")
