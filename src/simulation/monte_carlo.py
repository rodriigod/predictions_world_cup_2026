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
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.odds_tools import demargin_shin, power_vs_shin_gap
from src.data.wc_schema import build_match_features, match_features_frame

MAX_GOALS = 9          # truncamiento de la matriz de marcadores
DC_RHO = -0.08         # corrección Dixon-Coles para 0-0/1-1 (nota #4)
LAMBDA_JITTER = 0.10   # sigma del ruido lognormal por partido/iteración

# Temperatura de calibración aplicada a las lambdas ANTES de alimentar el
# Monte Carlo: lam' = mean * (lam/mean)^T. T<1 comprime hacia la media
# (menos sobreconfianza), T>1 la acentúa. El backtest multi-Mundial mostró
# que el modelo ya está bien calibrado (temperature scaling óptimo en T=1.0),
# así que el default es 1.0 = identidad; queda expuesto para re-calibrar.
LAMBDA_TEMPERATURE = 1.0

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


# ----------------------------- blend con mercado -----------------------------
def _demargin(odds_home: float, odds_draw: float,
              odds_away: float) -> tuple[float, float, float]:
    """Odds decimales -> probabilidades implícitas SIN margen (overround).
    prob = (1/odd) normalizado para que sumen 1 (método proporcional)."""
    imp = np.array([1.0 / odds_home, 1.0 / odds_draw, 1.0 / odds_away])
    return tuple(imp / imp.sum())


def blend_with_market(model_probs: pd.DataFrame, odds_csv, alpha: float = 0.3
                      ) -> pd.DataFrame:
    """Mezcla las probabilidades 1X2 del modelo con las del mercado.

    `model_probs`: DataFrame con columnas home_team, away_team, p_home,
    p_draw, p_away (1X2 del modelo, p.ej. derivadas de la matriz Dixon-Coles).
    `odds_csv`: ruta o DataFrame con home_team, away_team, odds_home,
    odds_draw, odds_away (odds decimales del mercado).
    `alpha`: PESO DEL MODELO. alpha=1.0 -> solo modelo, 0.0 -> solo mercado,
    0.5 -> mitad y mitad. El blend es un *log-linear opinion pool* (media
    geométrica ponderada renormalizada), el equivalente multiclase de mezclar
    en log-odds: p ∝ p_modelo^alpha · p_mercado^(1-alpha).

    Devuelve una copia de `model_probs` con p_home/p_draw/p_away mezcladas y
    una columna `blended` (True donde había odds del mercado)."""
    odds = pd.read_csv(odds_csv) if isinstance(odds_csv, (str, bytes)) else odds_csv
    out = model_probs.copy().reset_index(drop=True)
    out["blended"] = False
    if odds is None or len(odds) == 0:
        return out
    key = {(r.home_team, r.away_team): r for r in odds.itertuples()}
    eps = 1e-9
    flagged = []
    per_row_alpha = "alpha" in out.columns   # α dinámico por partido (opcional)
    for i, m in out.iterrows():
        row = key.get((m["home_team"], m["away_team"]))
        if row is None:
            continue
        oo = (float(row.odds_home), float(row.odds_draw), float(row.odds_away))
        try:
            mh, md, ma = demargin_shin(oo)     # Shin (1992): des-margining primario
        except (ValueError, ZeroDivisionError, TypeError):
            continue
        if not np.isfinite([mh, md, ma]).all():
            continue
        # cross-check con el método power: si difieren >1pp, línea desbalanceada
        gap = power_vs_shin_gap(oo)
        if gap > 1.0:
            flagged.append({"home_team": m["home_team"], "away_team": m["away_team"],
                            "max_divergence_pp": round(gap, 3)})
        a = float(m["alpha"]) if per_row_alpha else alpha
        pm = np.array([m["p_home"], m["p_draw"], m["p_away"]], float)
        pk = np.array([mh, md, ma], float)
        blended = (np.maximum(pm, eps) ** a) * (np.maximum(pk, eps) ** (1 - a))
        blended /= blended.sum()
        out.at[i, "p_home"], out.at[i, "p_draw"], out.at[i, "p_away"] = blended
        out.at[i, "blended"] = True
    if flagged:
        fp = Path(__file__).resolve().parents[2] / "results/flagged_odds.csv"
        fp.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(flagged).to_csv(fp, index=False)   # CAMBIO 1: log auditable
        names = [f"{f['home_team']}-{f['away_team']}" for f in flagged]
        print(f"      ⚠️ {len(flagged)} línea(s) desbalanceada(s) (Shin≠power "
              f">1pp) -> {fp.name}: {names[:6]}")
    return out


def get_dynamic_alpha(home_conf: str, away_conf: str, base_alpha: float = 0.3) -> float:
    """α (peso del modelo) variable por confederación — sesgo eurocéntrico del
    mercado (Cambio 2). Mercado UEFA/CONMEBOL muy eficiente -> α más bajo; un
    equipo potencialmente subvalorado (CONCACAF/CAF/AFC/OFC) vs uno fuerte ->
    α más alto. EXPERIMENTAL: NO validable (no hay odds históricas de Mundial)
    y la confederación ya salió neutra en backtests previos -> default OFF."""
    strong = {"UEFA", "CONMEBOL"}
    hs, aw = home_conf in strong, away_conf in strong
    if hs and aw:
        return max(base_alpha - 0.10, 0.05)
    if hs != aw:
        return min(base_alpha + 0.15, 0.55)
    return base_alpha


def lambdas_from_1x2(p1: float, pX: float, p2: float,
                     init: tuple[float, float] = (1.3, 1.1),
                     rho: float = DC_RHO) -> tuple[float, float]:
    """Resuelve (lam_a, lam_b) cuya matriz Dixon-Coles reproduce el 1X2 dado.
    El MC necesita λ para samplear marcadores; tras mezclar con el mercado en
    el espacio de probabilidades, re-derivamos las λ que las generan."""
    from scipy.optimize import minimize
    target = np.array([p1, pX, p2])

    def loss(z):
        la, lb = np.exp(z)            # exp -> λ siempre positivas
        m = _dixon_coles_matrix(min(la, 6.0), min(lb, 6.0), rho)
        pred = np.array([np.tril(m, -1).sum(), np.trace(m), np.triu(m, 1).sum()])
        return float(np.sum((pred - target) ** 2))

    res = minimize(loss, np.log(np.array(init)), method="Nelder-Mead",
                   options={"xatol": 1e-4, "fatol": 1e-10, "maxiter": 400})
    la, lb = np.exp(res.x)
    return float(np.clip(la, 0.15, 4.5)), float(np.clip(lb, 0.15, 4.5))


def dc_1x2(lam_a: float, lam_b: float, rho: float = DC_RHO
           ) -> tuple[float, float, float]:
    """Probabilidades 1X2 ANALÍTICAS derivadas directamente de la matriz
    Dixon-Coles (no muestreadas): P(1) = suma del triángulo inferior,
    P(X) = traza (todos los empates 0-0,1-1,...), P(2) = triángulo superior.
    Es la forma correcta de obtener P(empate) — sumando la diagonal de la
    matriz de marcadores — en vez de cualquier heurística."""
    m = _dixon_coles_matrix(lam_a, lam_b, rho)
    return (float(np.tril(m, -1).sum()), float(np.trace(m)),
            float(np.triu(m, 1).sum()))


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
                 calibration_temp: float = LAMBDA_TEMPERATURE,
                 odds_csv=None, blend_alpha: float = 1.0, dynamic_alpha: bool = False,
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
        self.calibration_temp = calibration_temp
        self.odds_csv = odds_csv
        self.blend_alpha = blend_alpha
        self.dynamic_alpha = dynamic_alpha
        self.n_blended = 0
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
        if self.odds_csv is not None and self.blend_alpha < 1.0:
            self._apply_market_blend()
        self._fixed = self._match_played(played_results)

    def _apply_market_blend(self) -> None:
        """Mezcla las λ del modelo con las odds del mercado: λ -> 1X2 (matriz
        DC) -> blend log-lineal con el mercado (alpha=peso del modelo) -> re-
        deriva λ' que reproducen el 1X2 mezclado. Solo afecta partidos que
        están en el CSV de odds; el resto queda con las λ del modelo."""
        rows = []
        for i in range(len(self.fixtures)):
            p1, pX, p2 = dc_1x2(self._lambdas[i, 0], self._lambdas[i, 1], self.rho)
            fx = self.fixtures.iloc[i]
            rows.append({"home_team": fx["team_a"], "away_team": fx["team_b"],
                         "p_home": p1, "p_draw": pX, "p_away": p2})
        model_probs = pd.DataFrame(rows)
        if self.dynamic_alpha and "confed" in self.teams.columns:
            cf = dict(zip(self.teams["team"], self.teams["confed"]))
            model_probs["alpha"] = [
                get_dynamic_alpha(cf.get(fx["team_a"], "UEFA"),
                                  cf.get(fx["team_b"], "UEFA"), self.blend_alpha)
                for _, fx in self.fixtures.iterrows()]
        blended = blend_with_market(model_probs, self.odds_csv, self.blend_alpha)
        for i in range(len(self.fixtures)):
            if not bool(blended.at[i, "blended"]):
                continue
            la, lb = lambdas_from_1x2(
                float(blended.at[i, "p_home"]), float(blended.at[i, "p_draw"]),
                float(blended.at[i, "p_away"]),
                init=(self._lambdas[i, 0], self._lambdas[i, 1]), rho=self.rho)
            self._lambdas[i] = (la, lb)
            self.n_blended += 1

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
        lams = self._calibrate(self.model.predict_lambda(
            match_features_frame(rows)))
        return lams.reshape(-1, 2)  # [n_fixtures, (lado_a, lado_b)]

    def _calibrate(self, lams: np.ndarray) -> np.ndarray:
        """Temperatura de calibración sobre las lambdas (T=1.0 = identidad).
        lam' = mean * (lam/mean)^T — comprime/acentúa la dispersión sin mover
        la media global de goles."""
        if self.calibration_temp == 1.0:
            return lams
        mean = float(np.mean(lams))
        return mean * (lams / mean) ** self.calibration_temp

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
        lams = self._calibrate(self.model.predict_lambda(
            match_features_frame(rows)))
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
    def run(self, n_sims: int = 50000, knockout: bool = False
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
        # Probabilidades 1X2 ANALÍTICAS de la matriz Dixon-Coles base (sin
        # jitter ni incentivos): P(empate) sale de la diagonal, no de heurística.
        dc = [dc_1x2(self._lambdas[i, 0], self._lambdas[i, 1], self.rho)
              for i in range(len(self.fixtures))]
        matches["p_win_a_dc"] = [round(p[0], 4) for p in dc]
        matches["p_draw_dc"] = [round(p[1], 4) for p in dc]
        matches["p_win_b_dc"] = [round(p[2], 4) for p in dc]
        matches["exp_goals_a"] = [t.goals_a / n_sims for t in tallies]
        matches["exp_goals_b"] = [t.goals_b / n_sims for t in tallies]
        most_likely = [t.scores.most_common(1)[0] for t in tallies]
        matches["most_likely_score"] = [f"{s[0][0]}-{s[0][1]}" for s in most_likely]
        matches["p_most_likely_score"] = [s[1] / n_sims for s in most_likely]
        # Marcador pronosticado: el que MAXIMIZA los puntos esperados de la
        # polla. Si predices un marcador s con resultado r, ganas 5 si es
        # exacto y 3 si solo aciertas el resultado, así que
        #   E[pts] = 5·P(s) + 3·(P(r) − P(s)) = 3·P(r) + 2·P(s).
        # Domina 3·P(r) -> el resultado (1X2) sale casi siempre el más
        # probable; el 2·P(s) afina el marcador DENTRO de ese resultado.
        # Mejor para la polla que redondear λ: recupera 1-0/2-0/0-0 reales.
        # En partidos ya jugados scores tiene solo el marcador real -> lo elige.
        def _best_score(t):
            pr = {"1": t.win_a / n_sims, "X": t.draw / n_sims,
                  "2": t.win_b / n_sims}
            best, best_e = (0, 0, "X", 0), -1.0
            for (a, b), c in t.scores.items():
                r = "1" if a > b else ("2" if a < b else "X")
                e = 3 * pr[r] + 2 * (c / n_sims)
                if e > best_e:
                    best, best_e = (a, b, r, c), e
            return best
        bests = [_best_score(t) for t in tallies]
        matches["pred_score"] = [f"{a}-{b}" for a, b, _r, _c in bests]
        matches["pred_result"] = [r for _a, _b, r, _c in bests]
        matches["p_pred_score"] = [c / n_sims for _a, _b, _r, c in bests]
        # Marcador alternativo por REDONDEO mitad-arriba de los goles esperados
        # (1.5->2, 1.4->1...). No es producción (nunca da 0); se guarda para
        # comparar contra el óptimo (ver README / scripts/microsim_groupstage).
        matches["pred_score_round"] = [
            f"{int(np.floor(t.goals_a / n_sims + 0.5))}-"
            f"{int(np.floor(t.goals_b / n_sims + 0.5))}" for t in tallies]
        matches["status"] = ["JUGADO" if i in self._fixed else "pendiente"
                             for i in range(len(self.fixtures))]
        return standings, matches
