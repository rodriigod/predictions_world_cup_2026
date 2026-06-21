"""Des-margining de odds 1X2 y consenso multi-casa.

Métodos para quitar el overround (margen) de odds decimales y recuperar
probabilidades "justas":
  - proportional : p_i = (1/odd_i) / Σ(1/odd)            (ingenuo, el actual)
  - shin         : Shin (1992) — corrige por insider/favorito-longshot
  - power        : Clarke et al. (2017) — p_i = (1/odd_i)^k con Σ=1

Y consenso entre varias casas en escala log (log-opinion pool), no promediando
probabilidades directas (Leitner-Zeileis-Hornik 2010).

Todos devuelven (p_home, p_draw, p_away) que suman 1.
"""
import numpy as np


def _inv(odds):
    return np.array([1.0 / o for o in odds], dtype=float)


def demargin_proportional(odds) -> np.ndarray:
    x = _inv(odds)
    return x / x.sum()


def demargin_power(odds) -> np.ndarray:
    """Resuelve k tal que Σ (1/odd_i)^k = 1. Como Σ(1/odd)>1, k>1 lo comprime."""
    from scipy.optimize import brentq
    x = _inv(odds)
    f = lambda k: np.sum(x ** k) - 1.0
    try:
        k = brentq(f, 1.0, 20.0, xtol=1e-9)
        p = x ** k
        return p / p.sum()
    except (ValueError, RuntimeError):
        return demargin_proportional(odds)


def demargin_shin(odds) -> np.ndarray:
    """Shin (1992): estima z (proporción de apuestas informadas) y recupera
    p_i = (sqrt(z² + 4(1−z)·x_i²/Σx) − z) / (2(1−z)), con z tal que Σp_i=1."""
    from scipy.optimize import brentq
    x = _inv(odds)
    bs = x.sum()

    def p_of(z):
        return (np.sqrt(z * z + 4 * (1 - z) * x * x / bs) - z) / (2 * (1 - z))

    f = lambda z: p_of(z).sum() - 1.0
    try:
        # f(0+) = sqrt(booksum) - 1 > 0 ; crece z -> baja la suma. Buscar raíz.
        z = brentq(f, 1e-9, 0.5, xtol=1e-9)
        p = p_of(z)
        return p / p.sum()
    except (ValueError, RuntimeError):
        return demargin_proportional(odds)


DEMARGIN = {"proportional": demargin_proportional,
            "shin": demargin_shin, "power": demargin_power}


def demargin(odds, method: str = "proportional") -> np.ndarray:
    return DEMARGIN[method](odds)


def power_vs_shin_gap(odds) -> float:
    """Máxima diferencia (pp) entre Shin y power — línea 'desbalanceada' si alta."""
    return float(np.max(np.abs(demargin_shin(odds) - demargin_power(odds))) * 100)


def logit_consensus(prob_rows) -> np.ndarray:
    """Consenso de varias casas en escala log (media geométrica renormalizada).
    `prob_rows`: lista de (p_home,p_draw,p_away) ya des-marginadas por casa."""
    arr = np.array(prob_rows, dtype=float)
    arr = np.clip(arr, 1e-9, 1.0)
    g = np.exp(np.mean(np.log(arr), axis=0))
    return g / g.sum()
