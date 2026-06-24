"""Motor de partido pre-partido (microsimulación 11v11).

Flujo NUEVO e independiente: no toca el pipeline histórico. Se corre a mano
antes de un partido, con las alineaciones reales, y deriva λ de las stats de
los 11 titulares (FBref) para simular el partido y blendear con la prob base.

Modelo simplificado: λ del equipo = xG agregado de los de campo ajustado por
el save% del portero rival y la intensidad de presión relativa. Sin física,
solo Poisson con ruido — coherente con el resto del repo.
"""
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

# ---- Fallbacks por posición (promedios de ligas top-5) ----
POSITION_DEFAULTS = {
    "GK": {"npxg_p90": 0.00, "progressive_carries_p90": 0.5,
           "pressures_p90": 2.0, "tackles_p90": 0.5,
           "aerials_won_pct": 0.60, "psxg_save_pct": 0.72},
    "CB": {"npxg_p90": 0.03, "progressive_carries_p90": 1.5,
           "pressures_p90": 8.0, "tackles_p90": 2.5,
           "aerials_won_pct": 0.60, "psxg_save_pct": 0.00},
    "FB": {"npxg_p90": 0.04, "progressive_carries_p90": 3.0,
           "pressures_p90": 9.0, "tackles_p90": 2.2,
           "aerials_won_pct": 0.50, "psxg_save_pct": 0.00},
    "MF": {"npxg_p90": 0.06, "progressive_carries_p90": 4.5,
           "pressures_p90": 10.0, "tackles_p90": 2.0,
           "aerials_won_pct": 0.45, "psxg_save_pct": 0.00},
    "AM": {"npxg_p90": 0.12, "progressive_carries_p90": 5.0,
           "pressures_p90": 7.0, "tackles_p90": 1.2,
           "aerials_won_pct": 0.40, "psxg_save_pct": 0.00},
    "FW": {"npxg_p90": 0.25, "progressive_carries_p90": 3.5,
           "pressures_p90": 5.0, "tackles_p90": 0.8,
           "aerials_won_pct": 0.45, "psxg_save_pct": 0.00},
    "WF": {"npxg_p90": 0.18, "progressive_carries_p90": 5.5,
           "pressures_p90": 6.0, "tackles_p90": 1.0,
           "aerials_won_pct": 0.38, "psxg_save_pct": 0.00},
}
POSITION_KEYS = set(POSITION_DEFAULTS)


@dataclass
class Player:
    name: str
    position: str
    npxg_p90: float = 0.0
    progressive_carries_p90: float = 0.0
    pressures_p90: float = 0.0
    tackles_p90: float = 0.0
    aerials_won_pct: float = 0.50
    psxg_save_pct: float = 0.72

    def __post_init__(self):
        pos = self.position if self.position in POSITION_DEFAULTS else "MF"
        d = POSITION_DEFAULTS[pos]
        # rellenar con el default de la posición lo que venga en 0 (sin dato)
        for stat, dv in d.items():
            if getattr(self, stat) in (0.0, None):
                setattr(self, stat, dv)


@dataclass
class Squad:
    team_name: str
    players: list = field(default_factory=list)

    @property
    def gk(self):
        gks = [p for p in self.players if p.position == "GK"]
        return gks[0] if gks else None

    @property
    def outfield(self):
        return [p for p in self.players if p.position != "GK"]

    def team_xg_p90(self) -> float:
        return sum(p.npxg_p90 for p in self.outfield)

    def team_pressure_p90(self) -> float:
        return sum(p.pressures_p90 for p in self.outfield)

    def team_tackle_p90(self) -> float:
        return sum(p.tackles_p90 for p in self.outfield)

    def gk_save_rate(self) -> float:
        return self.gk.psxg_save_pct if self.gk else 0.72

    def avg_aerial(self) -> float:
        defs_ = [p for p in self.outfield if p.position in ("CB", "FB")]
        return float(np.mean([p.aerials_won_pct for p in defs_])) if defs_ else 0.50


GK_BASELINE = 0.72   # save% de un portero promedio (referencia)
GK_SCALE = 1.2       # cuánto pesa un portero mejor/peor que el promedio


def squad_lambdas(home: Squad, away: Squad, is_neutral: bool = True
                  ) -> tuple[float, float]:
    """λ base de cada equipo a partir de las stats de jugadores.

    El xG agregado del equipo YA es la escala de goles esperados; el portero
    rival es un AJUSTE relativo a un portero promedio (no una reducción del
    ~70% como sugería la fórmula ingenua, que mandaba todo al piso). La presión
    del rival recorta un poco las ocasiones propias."""
    base_h, base_a = home.team_xg_p90(), away.team_xg_p90()
    # portero rival mejor que el promedio (0.72) -> menos goles; peor -> más
    base_h *= max(0.6, 1.0 - GK_SCALE * (away.gk_save_rate() - GK_BASELINE))
    base_a *= max(0.6, 1.0 - GK_SCALE * (home.gk_save_rate() - GK_BASELINE))
    ph, pa = home.team_pressure_p90(), away.team_pressure_p90()
    tot = ph + pa + 1e-6
    base_h *= 1.0 - 0.15 * (pa / tot - 0.5)   # presión del RIVAL reduce lo mío
    base_a *= 1.0 - 0.15 * (ph / tot - 0.5)
    if not is_neutral:
        base_h *= 1.10
    return float(np.clip(base_h, 0.4, 3.5)), float(np.clip(base_a, 0.4, 3.5))


def simulate_match(squad_home: Squad, squad_away: Squad, n_sims: int = 5000,
                   noise_sigma: float = 0.12, is_neutral: bool = True,
                   seed: int = 42) -> dict:
    """Simula el partido n_sims veces con las λ derivadas de los jugadores."""
    rng = np.random.default_rng(seed)
    lambda_home, lambda_away = squad_lambdas(squad_home, squad_away, is_neutral)

    noise_h = rng.lognormal(0, noise_sigma, n_sims)
    noise_a = rng.lognormal(0, noise_sigma, n_sims)
    gh = rng.poisson(lambda_home * noise_h)
    ga = rng.poisson(lambda_away * noise_a)

    p_home = float((gh > ga).mean())
    p_draw = float((gh == ga).mean())
    p_away = float((gh < ga).mean())
    score_counts = Counter(zip(gh.tolist(), ga.tolist()))
    score_matrix = {k: v / n_sims for k, v in score_counts.most_common(25)}
    return {
        "p_home_win": round(p_home, 4), "p_draw": round(p_draw, 4),
        "p_away_win": round(p_away, 4),
        "lambda_home": round(lambda_home, 3), "lambda_away": round(lambda_away, 3),
        "mean_goals_home": round(float(gh.mean()), 3),
        "mean_goals_away": round(float(ga.mean()), 3),
        "score_matrix": score_matrix,
        "most_likely_score": _consistent_modal(score_matrix,
                                               p_home, p_draw, p_away),
    }


def _consistent_modal(score_matrix: dict, p_home: float, p_draw: float,
                      p_away: float) -> tuple:
    """Marcador modal CONSISTENTE con el 1X2 (mismo criterio que
    monte_carlo.consistent_modal_score, pero sobre el dict de marcadores
    simulados): fija el resultado más probable y dentro de él toma el marcador
    más frecuente, evitando el empate fantasma."""
    outcome = int(np.argmax([p_home, p_draw, p_away]))   # 0 local,1 empate,2 visita
    def matches(k):
        gh, ga = k
        o = 0 if gh > ga else (1 if gh == ga else 2)
        return o == outcome
    cand = {k: v for k, v in score_matrix.items() if matches(k)}
    pool = cand or score_matrix
    return max(pool, key=pool.get)


def blend_predictions(p_statistical, p_micro, alpha: float = 0.65,
                      p_market=None, beta: float = 0.20) -> list:
    """Blend en log-odds (log-linear pool). alpha = peso del modelo estadístico
    vs microsimulación; si hay mercado, pesos = stat·alpha·(1-beta),
    micro·(1-alpha)·(1-beta), market·beta."""
    eps = 1e-6
    log_stat = np.log(np.array(p_statistical, float) + eps)
    log_micro = np.log(np.array(p_micro, float) + eps)
    if p_market is not None:
        mkt = np.array(p_market, float)
        mkt = mkt / mkt.sum()
        log_mkt = np.log(mkt + eps)
        log_blend = (alpha * (1 - beta) * log_stat
                     + (1 - alpha) * (1 - beta) * log_micro
                     + beta * log_mkt)
    else:
        log_blend = alpha * log_stat + (1 - alpha) * log_micro
    blend = np.exp(log_blend)
    return (blend / blend.sum()).tolist()
