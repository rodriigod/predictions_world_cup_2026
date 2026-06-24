"""Simulación Monte Carlo del torneo COMPLETO, agnóstica al modelo.

Toma un proveedor de matriz de marcadores `matrix(home, away)` (cualquiera de los
6 modelos: core/microsim/ensemble × sin/con datos) y simula grupos + eliminatorias
con el cuadro oficial 2026 (reusa core/simulation/knockout.py), devolviendo, por
equipo, la probabilidad de alcanzar cada ronda y de ser campeón — "todo lo que
debería pasar, hasta la final, con su porcentaje".

Grupos: 12 de 4; avanzan 1º, 2º y los 8 mejores 3º (formato 48). Cada partido se
resuelve muestreando un marcador de la matriz del modelo; en eliminatorias, el
empate se define por penales (sesgo leve al de mayor prob. de victoria del modelo).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from core.simulation.knockout import (KO_ROUNDS, ROUND_OF_32, allocate_thirds)

_THIRD_SLOTS = [mid for mid, _, sb, _ in ROUND_OF_32 if sb.startswith("3")]
from online_learning.panels import _variant
from online_learning.priors import to_es, canon

ROOT = Path(__file__).resolve().parents[1]
TEAMS = ROOT / "files/f0_raw/teams_2026.csv"
FIXTURES = ROOT / "files/f0_raw/fixtures_2026.csv"
ROUNDS = ["r32", "r16", "qf", "sf", "final", "champion"]
DUMMY_DATE = "2026-06-20"


class MatrixProvider:
    """matrix(home_es, away_es) -> np.ndarray, cacheada, para un modelo/régimen."""

    def __init__(self, model: str, regime: str):
        self.model, self.regime = model, regime
        from ensemble.predict import _context
        self.ctx = _context()
        self._cache: dict = {}

    def matrix(self, home: str, away: str) -> np.ndarray:
        key = (home, away)
        if key not in self._cache:
            v = _variant(self.ctx, home, away, DUMMY_DATE, self.model, self.regime)
            self._cache[key] = np.asarray(v["matrix"])
            self._cache[(away, home)] = self._cache[key].T   # simetría barata
        return self._cache[key]


def _sample_score(m: np.ndarray, rng) -> tuple[int, int]:
    flat = rng.choice(m.size, p=m.ravel() / m.sum())
    return divmod(flat, m.shape[1])


def _ko_winner(prov, a, b, rng) -> str:
    m = prov.matrix(a, b)
    i, j = _sample_score(m, rng)
    if i > j:
        return a
    if j > i:
        return b
    p_a = float(np.tril(m, -1).sum()); p_b = float(np.triu(m, 1).sum())
    return a if rng.random() < p_a / (p_a + p_b + 1e-12) else b


def _load_groups():
    teams = pd.read_csv(TEAMS)
    fx = pd.read_csv(FIXTURES)
    groups = {g: list(d["team"]) for g, d in teams.groupby("group")}
    fixtures = [(r.group, r.team_a, r.team_b) for r in fx.itertuples()]
    return groups, fixtures


def simulate(prov: MatrixProvider, *, n_sims: int = 1500, seed: int = 0
             ) -> pd.DataFrame:
    groups, fixtures = _load_groups()
    all_teams = [t for ts in groups.values() for t in ts]
    reach = {t: {r: 0 for r in ROUNDS} for t in all_teams}
    rng = np.random.default_rng(seed)

    for _ in range(n_sims):
        # ---- fase de grupos ----
        stats = {t: [0, 0, 0] for t in all_teams}       # pts, gf, ga
        for g, a, b in fixtures:
            ga, gb = _sample_score(prov.matrix(a, b), rng)
            stats[a][1] += ga; stats[a][2] += gb
            stats[b][1] += gb; stats[b][2] += ga
            if ga > gb:
                stats[a][0] += 3
            elif gb > ga:
                stats[b][0] += 3
            else:
                stats[a][0] += 1; stats[b][0] += 1

        slotmap, thirds = {}, []
        for g, ts in groups.items():
            order = sorted(ts, key=lambda t: (stats[t][0],
                           stats[t][1] - stats[t][2], stats[t][1], rng.random()),
                           reverse=True)
            slotmap["1" + g], slotmap["2" + g] = order[0], order[1]
            t3 = order[2]
            thirds.append((g, t3, stats[t3][0], stats[t3][1] - stats[t3][2],
                           stats[t3][1], rng.random()))
        thirds.sort(key=lambda x: x[2:], reverse=True)
        qualified = [x[0] for x in thirds[:8]]
        third_team = {g: t for g, t, *_ in thirds}
        try:
            alloc = allocate_thirds(qualified)
        except Exception:
            alloc = dict(zip(_THIRD_SLOTS, qualified))   # fallback robusto

        # marca clasificados a 16avos
        qualifiers = ([slotmap["1" + g] for g in groups]
                      + [slotmap["2" + g] for g in groups]
                      + [third_team[g] for g in qualified])
        for t in qualifiers:
            reach[t]["r32"] += 1

        # ---- eliminatorias ----
        winners, losers = {}, {}
        round_key = {"Dieciseisavos": "r16", "Octavos": "qf", "Cuartos": "sf",
                     "Semifinales": "final", "FINAL": "champion"}

        def resolve(slot, mid):
            if slot.startswith("W"):
                return winners[slot[1:]]
            if slot.startswith("L"):
                return losers[slot[1:]]
            if slot.startswith("3"):
                return third_team[alloc[mid]]
            return slotmap[slot]

        for round_name, mlist in KO_ROUNDS:
            if round_name == "Tercer puesto":
                for mid, sa, sb, _ in mlist:       # se juega pero no da ronda
                    ta, tb = resolve(sa, mid), resolve(sb, mid)
                    w = _ko_winner(prov, ta, tb, rng)
                    winners[str(mid)] = w
                    losers[str(mid)] = tb if w == ta else ta
                continue
            for mid, sa, sb, _ in mlist:
                ta, tb = resolve(sa, mid), resolve(sb, mid)
                w = _ko_winner(prov, ta, tb, rng)
                winners[str(mid)] = w
                losers[str(mid)] = tb if w == ta else ta
                if round_name in round_key:
                    reach[w][round_key[round_name]] += 1

    rows = [{"team": t, **{r: reach[t][r] / n_sims for r in ROUNDS}}
            for t in all_teams]
    df = pd.DataFrame(rows).sort_values("champion", ascending=False)
    return df.reset_index(drop=True)


def simulate_all(*, n_sims: int = 1500) -> dict:
    """{(model, regime): tabla por equipo} para los 6 modelos."""
    from online_learning.panels import MODELS, REGIMES
    out = {}
    for model in MODELS:
        for regime in REGIMES:
            prov = MatrixProvider(model, regime)
            out[(model, regime)] = simulate(prov, n_sims=n_sims)
    return out
