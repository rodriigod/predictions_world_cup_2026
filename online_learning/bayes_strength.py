"""C. Update Bayesiano de fuerza (Gamma-Poisson conjugado).

Modela la tasa de goles de cada equipo como Poisson(λ) con prior conjugado
Gamma sobre λ — uno ofensivo (goles marcados) y uno defensivo (goles recibidos):

    prior:      λ ~ Gamma(α0, β0)   con media α0/β0 = λ_core (estimación pre-torneo)
                y "fuerza de prior" κ = β0 = nº de partidos equivalentes del prior.
    posterior:  tras observar g_1..g_n,  α = α0 + Σ g_i,  β = κ + n.
                media posterior = α/β ;  intervalo de credibilidad 80% por cuantiles.

Con κ alto y pocos partidos jugados, el PRIOR HISTÓRICO PESA MUCHO: el Bayesiano
se mueve MENOS que el ELO puro. Eso es correcto y esperado — se documenta. El
output no es un punto fijo sino una posterior con su incertidumbre.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import gamma

from online_learning.dataset import load_results
from online_learning.priors import canon, pretournament_priors

# κ = partidos equivalentes del prior. Alto => el prior histórico domina con
# pocos partidos jugados (decisión deliberada del encargo).
PRIOR_KAPPA = 5.0
CRED = 0.80                  # intervalo de credibilidad


def _observed_goals():
    """{equipo_en: (lista_goles_a_favor, lista_goles_en_contra)} de los 2026."""
    gf: dict[str, list] = {}
    ga: dict[str, list] = {}
    for r in load_results().itertuples():
        a, b = r.home_team, r.away_team
        gh, ag = int(r.home_goals), int(r.away_goals)
        gf.setdefault(a, []).append(gh); ga.setdefault(a, []).append(ag)
        gf.setdefault(b, []).append(ag); ga.setdefault(b, []).append(gh)
    return gf, ga


def _posterior(prior_mean: float, goals: list, kappa: float):
    """Devuelve (alpha, beta, media, lo, hi) de la posterior Gamma-Poisson."""
    a0 = prior_mean * kappa
    alpha = a0 + float(np.sum(goals))
    beta = kappa + len(goals)
    mean = alpha / beta
    lo, hi = gamma.ppf([(1 - CRED) / 2, 1 - (1 - CRED) / 2], alpha, scale=1 / beta)
    return alpha, beta, float(mean), float(lo), float(hi)


def compute_posteriors(*, kappa: float = PRIOR_KAPPA) -> dict:
    """{equipo_en: {off:{mean,lo,hi,n}, def:{mean,lo,hi,n}}} con la posterior de
    λ ofensivo (goles marcados) y defensivo (goles recibidos)."""
    priors = pretournament_priors()
    gf, ga = _observed_goals()
    out = {}
    for en, p in priors.items():
        og, dg = gf.get(en, []), ga.get(en, [])
        _, _, m_o, lo_o, hi_o = _posterior(p["lam_off"], og, kappa)
        _, _, m_d, lo_d, hi_d = _posterior(p["lam_def"], dg, kappa)
        out[en] = {
            "off": {"prior": p["lam_off"], "mean": m_o, "lo": lo_o, "hi": hi_o,
                    "n": len(og)},
            "def": {"prior": p["lam_def"], "mean": m_d, "lo": lo_d, "hi": hi_d,
                    "n": len(dg)}}
    return out


_CACHE = {"key": None, "post": None}


def _state():
    from online_learning.dataset import RESULTS_CSV
    key = RESULTS_CSV.stat().st_mtime if RESULTS_CSV.exists() else 0
    if _CACHE["key"] != key:
        _CACHE["post"] = compute_posteriors()
        _CACHE["key"] = key
    return _CACHE["post"]


def get_strength_posterior(team: str) -> dict:
    """Posterior de fuerza ofensiva/defensiva de `team`: media + IC80%."""
    return _state()[canon(team)]
