"""Simulación de Monte Carlo de la fase de grupos del Mundial 2026.

Formato FIFA 2026: 48 equipos, 12 grupos de 4. Avanzan a 16avos los dos
primeros de cada grupo + los 8 mejores terceros.

El fixture real (fechas y emparejamientos de la polla mundialera) se
carga desde files/f0_raw/fixtures_2026.csv. Cada iteración del torneo:

1. Perturba las lambdas base con ruido lognormal (sigma=LAMBDA_JITTER):
   representa las condiciones del día del partido — clima/calor, estado
   del campo, rendimiento puntual — que el modelo no puede conocer.
2. Juega las fechas 1 y 2; antes de la fecha 3 aplica los incentivos de
   la Categoría 4 del diccionario (rotación de clasificados, empate que
   sirve a ambos, necesidad de ganar) según el estado REAL del grupo en
   esa iteración.
3. Acumula estadísticas por equipo (clasificación) y POR PARTIDO
   (1X2, marcador más probable, goles esperados), de modo que cada uno
   de los 72 partidos queda simulado n_sims veces.
"""

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.data.wc_schema import build_match_features, match_features_frame

MAX_GOALS = 9          # truncamiento de la matriz de marcadores
DC_RHO = -0.08         # corrección Dixon-Coles para 0-0/1-1 (nota #4)
LAMBDA_JITTER = 0.10   # sigma del ruido lognormal por partido/iteración

# Multiplicadores de incentivo (Categoría 4 del diccionario)
ROTATION_ATTACK = 0.88     # clasificado con 6 pts rota: ataca menos
ROTATION_CONCEDE = 1.08    # ...y concede más
MUST_WIN_ATTACK = 1.06     # obligado a ganar: más vértigo ofensivo
MUST_WIN_CONCEDE = 1.04    # ...y más espacios atrás
CAGEY_DRAW = 0.92          # empate sirve a ambos: partido cerrado

# Resolución de eliminatorias tras empate en 90' (prórroga + penales).
# La evidencia (tandas de penales ≈ moneda al aire) recomienda NO decidir
# por la fuerza completa de 90', sino dar a la prórroga otra oportunidad y
# luego un sesgo leve al favorito en los penales.
KO_ET_FRACTION = 0.33      # 30' de prórroga ≈ 1/3 del tiempo de 90'
KO_PEN_BIAS = 0.05         # sesgo máximo del favorito en penales (~±5%)
KO_PEN_ELO_SCALE = 200.0   # cuán rápido el ELO inclina la tanda


def _dixon_coles_matrix(lam_a: float, lam_b: float,
                        rho: float = DC_RHO) -> np.ndarray:
    """Matriz P(marcador a, b) Poisson independiente + corrección tau."""
    goals = np.arange(MAX_GOALS + 1)
    log_fact = np.cumsum(np.concatenate(([0.0], np.log(goals[1:]))))
    pa = np.exp(goals * np.log(lam_a) - lam_a - log_fact)
    pb = np.exp(goals * np.log(lam_b) - lam_b - log_fact)
    m = np.outer(pa, pb)
    m[0, 0] *= 1 - lam_a * lam_b * rho
    m[0, 1] *= 1 + lam_a * rho
    m[1, 0] *= 1 + lam_b * rho
    m[1, 1] *= 1 - rho
    return m / m.sum()


def _sample_score(matrix: np.ndarray, rng: np.random.Generator) -> tuple[int, int]:
    flat = rng.choice(matrix.size, p=matrix.ravel())
    return divmod(flat, matrix.shape[1])


def _modal_score_given_result(tally: "_MatchTally",
                              result: str) -> tuple[tuple[int, int], int]:
    """Marcador más frecuente entre las simulaciones con ese 1X2."""
    if result == "1":
        ok = lambda s: s[0] > s[1]
    elif result == "2":
        ok = lambda s: s[0] < s[1]
    else:
        ok = lambda s: s[0] == s[1]
    candidates = [(s, c) for s, c in tally.scores.items() if ok(s)]
    return max(candidates, key=lambda x: x[1])


@dataclass
class _Standing:
    points: int = 0
    gf: int = 0
    ga: int = 0
    results: dict = field(default_factory=dict)  # rival -> puntos h2h

    @property
    def gd(self) -> int:
        return self.gf - self.ga


@dataclass
class _MatchTally:
    """Acumulador de resultados de UN partido a través de las n_sims."""
    win_a: int = 0
    draw: int = 0
    win_b: int = 0
    goals_a: int = 0
    goals_b: int = 0
    scores: Counter = field(default_factory=Counter)

    def add(self, ga: int, gb: int) -> None:
        if ga > gb:
            self.win_a += 1
        elif ga < gb:
            self.win_b += 1
        else:
            self.draw += 1
        self.goals_a += ga
        self.goals_b += gb
        self.scores[(ga, gb)] += 1


class GroupStageSimulator:
    def __init__(self, teams: pd.DataFrame, fixtures: pd.DataFrame, model,
                 rho: float = DC_RHO, lambda_jitter: float = LAMBDA_JITTER,
                 played_results: pd.DataFrame | None = None,
                 seed: int = 42):
        """teams: TEAM_COLUMNS, una fila por equipo.
        fixtures: columnas group, matchday, date, time, team_a, team_b.
        played_results: partidos ya jugados (team_a, team_b, goals_a,
        goals_b) — quedan FIJOS en todas las iteraciones."""
        self.teams = teams.reset_index(drop=True)
        self.model = model
        self.rho = rho
        self.lambda_jitter = lambda_jitter
        self.rng = np.random.default_rng(seed)
        self._idx = {t: i for i, t in enumerate(self.teams["team"])}

        unknown = (set(fixtures["team_a"]) | set(fixtures["team_b"])) - set(self._idx)
        if unknown:
            raise ValueError(f"Equipos del fixture sin datos: {unknown}")
        self.fixtures = (fixtures.sort_values(["group", "matchday"])
                         .reset_index(drop=True))
        self.groups = {g: sorted(set(gdf["team_a"]) | set(gdf["team_b"]))
                       for g, gdf in self.fixtures.groupby("group")}
        for g, names in self.groups.items():
            if len(names) != 4:
                raise ValueError(f"El grupo {g} tiene {len(names)} equipos")
        self._lambdas = self._precompute_lambdas()
        self._fixed = self._match_played(played_results)

    def _match_played(self, played: pd.DataFrame | None) -> dict:
        """Mapea resultados jugados a índices del fixture (cualquier
        orientación de los equipos)."""
        if played is None or played.empty:
            return {}
        fixed = {}
        for p in played.itertuples():
            for i, fx in enumerate(self.fixtures.itertuples()):
                if (fx.team_a, fx.team_b) == (p.team_a, p.team_b):
                    fixed[i] = (int(p.goals_a), int(p.goals_b))
                    break
                if (fx.team_a, fx.team_b) == (p.team_b, p.team_a):
                    fixed[i] = (int(p.goals_b), int(p.goals_a))
                    break
            else:
                raise ValueError(
                    f"Resultado jugado sin fixture: {p.team_a}-{p.team_b}")
        return fixed

    # ---------- lambdas base ----------
    def _precompute_lambdas(self) -> np.ndarray:
        """lambda base (lado a y b) de cada fixture, vía el modelo en batch.
        Incluye la diferencia REAL de días de descanso según las fechas
        del fixture (en J1 todos llegan parejos)."""
        last_date: dict[str, pd.Timestamp] = {}
        rest: list[float] = []
        for fx in self.fixtures.sort_values("date").itertuples():
            d = pd.Timestamp(fx.date)
            ra = (d - last_date[fx.team_a]).days if fx.team_a in last_date else 7
            rb = (d - last_date[fx.team_b]).days if fx.team_b in last_date else 7
            rest.append(ra - rb)
            last_date[fx.team_a] = d
            last_date[fx.team_b] = d
        rest_by_pos = pd.Series(
            rest, index=self.fixtures.sort_values("date").index)

        rows = []
        for i, fx in enumerate(self.fixtures.itertuples()):
            a = self.teams.loc[self._idx[fx.team_a]]
            b = self.teams.loc[self._idx[fx.team_b]]
            rd = float(rest_by_pos.loc[i])
            rows.append(build_match_features(a, b, fx.matchday, rd))
            rows.append(build_match_features(b, a, fx.matchday, -rd))
        lams = self.model.predict_lambda(match_features_frame(rows))
        return lams.reshape(-1, 2)  # [n_fixtures, (lado_a, lado_b)]

    # ---------- incentivos fecha 3 ----------
    @staticmethod
    def _incentive_multipliers(pts_own: int, pts_opp: int) -> tuple[float, float]:
        """(mult ataque propio, mult ataque rival inducido) en fecha 3."""
        atk, induce = 1.0, 1.0
        if pts_own == 6:                      # clasificado seguro: rota
            atk *= ROTATION_ATTACK
            induce *= ROTATION_CONCEDE
        elif pts_own <= 1:                    # al borde: a ganar o ganar
            atk *= MUST_WIN_ATTACK
            induce *= MUST_WIN_CONCEDE
        if 3 <= pts_own <= 4 and 3 <= pts_opp <= 4:   # empate cómodo p/ambos
            atk *= CAGEY_DRAW
        return atk, induce

    # ---------- ordenamientos ----------
    def _rank_group(self, names: list[str], st: dict) -> list[str]:
        """Criterios FIFA: pts, DG, GF, h2h entre empatados, sorteo."""
        def key(t):
            return (st[t].points, st[t].gd, st[t].gf)
        ordered = sorted(names, key=key, reverse=True)
        final, k = [], 0
        while k < len(ordered):
            block = [ordered[k]]
            while (k + len(block) < len(ordered)
                   and key(ordered[k + len(block)]) == key(block[0])):
                block.append(ordered[k + len(block)])
            if len(block) > 1:
                self.rng.shuffle(block)  # sorteo como último recurso
                block.sort(key=lambda t: sum(
                    st[t].results.get(o, 0) for o in block if o != t),
                    reverse=True)
            final.extend(block)
            k += len(block)
        return final

    def _rank_thirds(self, thirds: list[tuple[str, str, _Standing]]
                     ) -> list[tuple[str, str]]:
        """thirds: (equipo, grupo, standing) -> [(equipo, grupo)] ranked."""
        arr = sorted(
            thirds,
            key=lambda t: (t[2].points, t[2].gd, t[2].gf, self.rng.random()),
            reverse=True)
        return [(t, g) for t, g, _ in arr]

    # ---------- lambdas de cualquier cruce (para eliminatorias) ----------
    def _pairwise_lambdas(self) -> np.ndarray:
        """Matriz [i, j] = lambda del equipo i contra el equipo j en
        cancha neutral de eliminación directa (jornada 0)."""
        n = len(self.teams)
        rows, idx = [], []
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                rows.append(build_match_features(
                    self.teams.loc[i], self.teams.loc[j], 0))
                idx.append((i, j))
        lams = self.model.predict_lambda(match_features_frame(rows))
        out = np.full((n, n), np.nan)
        for (i, j), l in zip(idx, lams):
            out[i, j] = l
        return out

    def _play_ko_match(self, ia: int, ib: int, pair_lams: np.ndarray
                       ) -> tuple[int, int]:
        """(ganador, perdedor) de un cruce eliminatorio. Empate en 90' ->
        prórroga (lambda reducida); si sigue empatado -> penales, que son
        casi una moneda al aire con un sesgo leve al de más ELO."""
        jit = np.exp(self.rng.normal(0.0, self.lambda_jitter, 2))
        la = pair_lams[ia, ib] * jit[0]
        lb = pair_lams[ib, ia] * jit[1]
        ga_, gb_ = _sample_score(_dixon_coles_matrix(la, lb, self.rho),
                                 self.rng)
        if ga_ != gb_:
            return (ia, ib) if ga_ > gb_ else (ib, ia)
        # prórroga: 30' ≈ un tercio de los goles esperados
        ea, eb = _sample_score(
            _dixon_coles_matrix(la * KO_ET_FRACTION, lb * KO_ET_FRACTION,
                                self.rho), self.rng)
        if ea != eb:
            return (ia, ib) if ea > eb else (ib, ia)
        # penales: ~50%, leve ventaja al favorito por ELO
        elo_diff = self.teams["elo"].iat[ia] - self.teams["elo"].iat[ib]
        p_pen_a = 0.5 + KO_PEN_BIAS * np.tanh(elo_diff / KO_PEN_ELO_SCALE)
        return (ia, ib) if self.rng.random() < p_pen_a else (ib, ia)

    # ---------- torneo completo ----------
    def run(self, n_sims: int = 10000, knockout: bool = False
            ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Devuelve (tabla de clasificación, tabla de partidos).

        Con knockout=True juega también 16avos→final en cada iteración
        y agrega columnas p_r16/p_qf/p_sf/p_final/p_champion."""
        from .knockout import KO_ROUNDS, allocate_thirds

        n = len(self.teams)
        win_group = np.zeros(n); runner_up = np.zeros(n)
        third = np.zeros(n); third_adv = np.zeros(n)
        advance = np.zeros(n); points_sum = np.zeros(n)
        gf_sum = np.zeros(n); ga_sum = np.zeros(n)
        tallies = [_MatchTally() for _ in range(len(self.fixtures))]
        ko_reach = {r: np.zeros(n) for r in
                    ["r16", "qf", "sf", "final", "champion"]}
        pair_lams = self._pairwise_lambdas() if knockout else None

        fx_by_group: dict[str, list[tuple[int, int, str, str]]] = {}
        for i, fx in enumerate(self.fixtures.itertuples()):
            fx_by_group.setdefault(fx.group, []).append(
                (i, fx.matchday, fx.team_a, fx.team_b))

        for _ in range(n_sims):
            # ruido del día del partido (clima, físico, campo) por lado
            jitter = np.exp(self.rng.normal(
                0.0, self.lambda_jitter, self._lambdas.shape))
            lams = self._lambdas * jitter

            thirds_pool = []
            slotmap: dict[str, int] = {}
            for g, matches in fx_by_group.items():
                st = {t: _Standing() for t in self.groups[g]}
                for i, md, ta, tb in matches:
                    if i in self._fixed:           # partido ya jugado
                        ga_, gb_ = self._fixed[i]
                    else:
                        lam_a, lam_b = lams[i]
                        if md == 3:
                            ma, ind_a = self._incentive_multipliers(
                                st[ta].points, st[tb].points)
                            mb, ind_b = self._incentive_multipliers(
                                st[tb].points, st[ta].points)
                            lam_a *= ma * ind_b
                            lam_b *= mb * ind_a
                        ga_, gb_ = _sample_score(
                            _dixon_coles_matrix(lam_a, lam_b, self.rho),
                            self.rng)
                    tallies[i].add(ga_, gb_)
                    st[ta].gf += ga_; st[ta].ga += gb_
                    st[tb].gf += gb_; st[tb].ga += ga_
                    pa = 3 if ga_ > gb_ else (1 if ga_ == gb_ else 0)
                    pb = 3 if gb_ > ga_ else (1 if ga_ == gb_ else 0)
                    st[ta].points += pa; st[tb].points += pb
                    st[ta].results[tb] = pa; st[tb].results[ta] = pb

                order = self._rank_group(self.groups[g], st)
                for pos, t in enumerate(order):
                    ti = self._idx[t]
                    points_sum[ti] += st[t].points
                    gf_sum[ti] += st[t].gf
                    ga_sum[ti] += st[t].ga
                    if pos == 0:
                        win_group[ti] += 1; advance[ti] += 1
                        slotmap["1" + g] = ti
                    elif pos == 1:
                        runner_up[ti] += 1; advance[ti] += 1
                        slotmap["2" + g] = ti
                    elif pos == 2:
                        third[ti] += 1
                        thirds_pool.append((t, g, st[t]))
            ranked_thirds = self._rank_thirds(thirds_pool)
            third_by_group: dict[str, int] = {}
            for t, g in ranked_thirds[:8]:
                ti = self._idx[t]
                third_adv[ti] += 1
                advance[ti] += 1
                third_by_group[g] = ti

            if knockout:
                alloc = allocate_thirds(list(third_by_group))
                winners: dict[str, int] = {}
                losers: dict[str, int] = {}

                def resolve(slot: str, mid: int) -> int:
                    if slot.startswith("W"):
                        return winners[slot[1:]]
                    if slot.startswith("L"):
                        return losers[slot[1:]]
                    if slot.startswith("3"):
                        return third_by_group[alloc[mid]]
                    return slotmap[slot]

                reach_round = {"Octavos": "r16", "Cuartos": "qf",
                               "Semifinales": "sf", "FINAL": "final"}
                for round_name, matches_list in KO_ROUNDS:
                    if round_name == "Tercer puesto":
                        continue
                    for mid, sa, sb, _date in matches_list:
                        ia, ib = resolve(sa, mid), resolve(sb, mid)
                        wi, li = self._play_ko_match(ia, ib, pair_lams)
                        winners[str(mid)] = wi
                        losers[str(mid)] = li
                        if round_name in reach_round:
                            ko_reach[reach_round[round_name]][[ia, ib]] += 1
                ko_reach["champion"][winners["104"]] += 1

        standings = self.teams[["team", "group", "elo"]].copy()
        standings["p_win_group"] = win_group / n_sims
        standings["p_runner_up"] = runner_up / n_sims
        standings["p_third_place"] = third / n_sims
        standings["p_advance_as_third"] = third_adv / n_sims
        standings["p_advance"] = advance / n_sims
        standings["exp_points"] = points_sum / n_sims
        standings["exp_gf"] = gf_sum / n_sims
        standings["exp_ga"] = ga_sum / n_sims
        if knockout:
            for r, arr in ko_reach.items():
                standings[f"p_{r}"] = arr / n_sims
        standings = (standings
                     .sort_values(["group", "p_advance", "exp_points"],
                                  ascending=[True, False, False])
                     .reset_index(drop=True))

        matches = self.fixtures.copy()
        matches["lambda_a"] = self._lambdas[:, 0].round(3)
        matches["lambda_b"] = self._lambdas[:, 1].round(3)
        matches["p_win_a"] = [t.win_a / n_sims for t in tallies]
        matches["p_draw"] = [t.draw / n_sims for t in tallies]
        matches["p_win_b"] = [t.win_b / n_sims for t in tallies]
        matches["exp_goals_a"] = [t.goals_a / n_sims for t in tallies]
        matches["exp_goals_b"] = [t.goals_b / n_sims for t in tallies]
        most_likely = [t.scores.most_common(1)[0] for t in tallies]
        matches["most_likely_score"] = [f"{s[0][0]}-{s[0][1]}" for s in most_likely]
        matches["p_most_likely_score"] = [s[1] / n_sims for s in most_likely]
        # resultado 1X2 más probable de cada partido
        def _most_likely(t):
            return ("1" if t.win_a >= max(t.draw, t.win_b)
                    else ("X" if t.draw >= t.win_b else "2"))
        # "Empate técnico": cuando ganar le es casi igual de probable a
        # ambos lados (|P(1) - P(2)| <= DRAW_MARGIN_PP puntos porcentuales,
        # medidos como se muestran en el reporte), el partido se marca como
        # empate aunque haya un favorito leve. No aplica a partidos ya
        # jugados (resultado real fijo).
        DRAW_MARGIN_PP = 3
        pred_result = []
        for i, t in enumerate(tallies):
            r = _most_likely(t)
            if i not in self._fixed and r != "X":
                diff_pp = abs(round(100 * t.win_a / n_sims)
                              - round(100 * t.win_b / n_sims))
                if diff_pp <= DRAW_MARGIN_PP:
                    r = "X"
            pred_result.append(r)
        matches["pred_result"] = pred_result
        # marcador pronosticado: el más frecuente CONDICIONADO al
        # resultado 1X2 pronosticado (consistente para llenar la polla)
        pred_scores = [
            _modal_score_given_result(t, r)
            for t, r in zip(tallies, matches["pred_result"])]
        matches["pred_score"] = [f"{s[0]}-{s[1]}" for s, _ in pred_scores]
        matches["p_pred_score"] = [c / n_sims for _, c in pred_scores]
        matches["status"] = ["JUGADO" if i in self._fixed else "pendiente"
                             for i in range(len(self.fixtures))]
        return standings, matches
