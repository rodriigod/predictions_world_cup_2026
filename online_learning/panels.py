"""Paneles comparativos por partido: 6 modelos (core/microsim/ensemble × sin/con
datos 2026), cada uno con su matriz de marcadores 0-5×0-5, barras de resultado
1X2 y top marcadores — al estilo de las imágenes pedidas.

"sin datos"  = pipeline de producción (ratings pre-torneo).
"con datos"  = mismos modelos pero con los ratings online-updated (online_learning).

Depende de online_learning (no de producción): es una herramienta de comparación.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from core.simulation.monte_carlo import (DC_RHO, _dixon_coles_matrix, dc_1x2)
from core.data.wc_schema import build_match_features, match_features_frame
from online_learning.predict_updated import (core_lambdas_updated, micro_updated)
from online_learning.priors import canon, to_es

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "results/match_panels"
SHOW = 6                       # marcadores 0..5 en la matriz mostrada

MODELS = ["core", "microsim", "ensemble"]
REGIMES = ["sin", "con"]
LABELS = {"core": "core (Poisson/Dixon-Coles)",
          "microsim": "microsim (fuerza de plantel)",
          "ensemble": "ensemble (stacking, LLM incl.)"}
REGIME_LABEL = {"sin": "SIN datos 2026 (pre-torneo)",
                "con": "CON datos 2026 (online-updated)"}


# --------------------------- cálculo de variantes ---------------------------
def _core_lambdas_sin(ctx, home, away):
    sa = ctx._series(home, float(ctx.is_host.get(home, 0.0)))
    sb = ctx._series(away, 0.0)
    lam = ctx.poisson.predict_lambda(match_features_frame(
        [build_match_features(sa, sb, 1, 0.0),
         build_match_features(sb, sa, 1, 0.0)]))
    return float(lam[0]), float(lam[1])


def _variant(ctx, home, away, date, model, regime) -> dict:
    """Devuelve {probs:(1,X,2), lam_h, lam_a, matrix} de un modelo/régimen."""
    if model == "core":
        if regime == "sin":
            lh, la = _core_lambdas_sin(ctx, home, away)
        else:
            lh, la = core_lambdas_updated(ctx, home, away)
        m = _dixon_coles_matrix(lh, la, DC_RHO)
        return {"probs": dc_1x2(lh, la), "lam_h": lh, "lam_a": la, "matrix": m}

    if model == "microsim":
        mic = ctx.micro if regime == "sin" else micro_updated(ctx)
        h_es, a_es = to_es(canon(home)), to_es(canon(away))
        lh, la = mic.lambdas(h_es, a_es, neutral=True)
        m = _dixon_coles_matrix(lh, la, mic.rho)
        return {"probs": dc_1x2(lh, la), "lam_h": lh, "lam_a": la, "matrix": m}

    # ensemble
    if regime == "sin":
        from ensemble.predict import predict_final
        p = predict_final(home, away, date, use_llm=False)
    else:
        from online_learning.predict_updated import predict_final_updated
        p = predict_final_updated(home, away, date)
    return {"probs": (p.prob_home, p.prob_draw, p.prob_away),
            "lam_h": p.lambda_home, "lam_a": p.lambda_away,
            "matrix": np.asarray(p.score_matrix)}


def compute_all(home, away, date) -> dict:
    """{(model, regime): variant} para los 6 paneles."""
    from ensemble.predict import _context
    ctx = _context()
    out = {}
    for model in MODELS:
        for regime in REGIMES:
            out[(model, regime)] = _variant(ctx, home, away, date, model, regime)
    return out


# ------------------------------- ploteo -------------------------------------
def _abbr(name: str) -> str:
    return re.sub(r"[^A-Za-zÁÉÍÓÚÑ]", "", name).upper()[:3] or "EQ"


def plot_panel(home, away, model, regime, v, outpath: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    ha, aa = _abbr(home), _abbr(away)
    M = np.asarray(v["matrix"])[:SHOW, :SHOW] * 100.0   # % 0..5
    p1, pX, p2 = v["probs"]
    bi, bj = np.unravel_index(int(np.argmax(np.asarray(v["matrix"]))),
                              np.asarray(v["matrix"]).shape)

    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1.25, 1], height_ratios=[1, 1],
                           hspace=0.45, wspace=0.28)

    # --- matriz de marcadores ---
    axm = fig.add_subplot(gs[:, 0])
    im = axm.imshow(M, cmap="YlOrRd", aspect="auto")
    for i in range(SHOW):
        for j in range(SHOW):
            axm.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center",
                     fontsize=8, color="black")
    if bi < SHOW and bj < SHOW:                      # casilla más probable
        axm.add_patch(plt.Rectangle((bj - .5, bi - .5), 1, 1, fill=False,
                                    edgecolor="#1565C0", lw=2.5))
    axm.set_xticks(range(SHOW)); axm.set_yticks(range(SHOW))
    axm.set_xlabel(f"Goles {away}"); axm.set_ylabel(f"Goles {home}")
    axm.set_title("Probabilidad por marcador (%)")
    fig.colorbar(im, ax=axm, fraction=0.046, pad=0.04)

    # --- barras de resultado ---
    axr = fig.add_subplot(gs[0, 1])
    cols = ["#2E7D32", "#9E9E9E", "#1565C0"]
    bars = axr.bar([f"{home}\ngana", "Empate", f"{away}\ngana"],
                   [p1 * 100, pX * 100, p2 * 100], color=cols)
    for b, val in zip(bars, [p1, pX, p2]):
        axr.text(b.get_x() + b.get_width() / 2, b.get_height() + 1,
                 f"{val*100:.1f}%", ha="center", fontsize=9, fontweight="bold")
    axr.set_ylim(0, 100); axr.set_ylabel("Prob. (%)")
    axr.set_title("Probabilidad de resultado")

    # --- top-10 marcadores ---
    axt = fig.add_subplot(gs[1, 1])
    full = np.asarray(v["matrix"])
    flat = [(i, j, full[i, j]) for i in range(full.shape[0])
            for j in range(full.shape[1])]
    flat.sort(key=lambda t: t[2], reverse=True)
    top = flat[:10][::-1]
    labels = [f"{ha} {i}-{j} {aa}" for i, j, _ in top]
    vals = [p * 100 for _, _, p in top]
    bcol = ["#2E7D32" if i > j else ("#9E9E9E" if i == j else "#1565C0")
            for i, j, _ in top]
    axt.barh(labels, vals, color=bcol)
    for y, val in enumerate(vals):
        axt.text(val + 0.1, y, f"{val:.1f}", va="center", fontsize=7)
    axt.set_xlabel("Prob. (%)"); axt.set_title("Top 10 marcadores")
    axt.tick_params(axis="y", labelsize=7)

    fig.suptitle(f"{home} vs {away}  —  {LABELS[model]}\n{REGIME_LABEL[regime]}",
                 fontsize=12, fontweight="bold")
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")


def generate_match(home, away, date, *, outdir: Path = OUTDIR) -> list[Path]:
    """Genera los 6 PNG de un partido. Devuelve las rutas escritas."""
    variants = compute_all(home, away, date)
    folder = outdir / f"{_safe(home)}_vs_{_safe(away)}"
    paths = []
    for (model, regime), v in variants.items():
        out = folder / f"{model}_{regime}_datos.png"
        plot_panel(home, away, model, regime, v, out)
        paths.append(out)
    return paths
