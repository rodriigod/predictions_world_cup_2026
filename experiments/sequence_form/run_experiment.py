#!/usr/bin/env python3
"""D. Experimento: ¿una secuencia de forma APRENDIDA (GRU) mejora a los pi-ratings?

Compara, con el MISMO TimeSeriesSplit temporal (entrena en el pasado, evalúa en el
futuro) y las mismas métricas que core/ (RPS + accuracy), tres formas de
representar la "fuerza/forma" de cada selección como input a un clasificador 1X2:

  1. pi-ratings solo      : ataque/defensa pi + ΔElo (lo que core/ ya usa) -> logística.
  2. secuencia aprendida  : un GRU pequeño codifica los últimos N partidos de cada
     selección en un vector de "forma" -> capa lineal -> 1X2.
  3. ambas                : forma del GRU + pi-ratings concatenados -> 1X2.

NO reemplaza Dixon-Coles ni el pipeline de core/: es una comparación AISLADA del
input de fuerza de equipo, como pide el encargo. El veredicto se reporta tal cual
salga (si la secuencia NO gana a pi-ratings, se dice).

Uso: python experiments/sequence_form/run_experiment.py [--n-splits 4 --epochs 12]
Salida: results/reports/sequence_form_experiment.md
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from core.utils.metrics import ModelMetrics
from experiments.sequence_form.build_dataset import build

REPORT = ROOT / "results/reports/sequence_form_experiment.md"
SEED = 0


def _metrics(y, proba) -> dict:
    proba = np.asarray(proba, float)
    return {"acc": float((proba.argmax(1) == y).mean()),
            "rps": ModelMetrics.rps(y, proba)}


# ------------------------------- GRU (torch) --------------------------------
def _make_net(F, H, pi_dim):
    import torch
    import torch.nn as nn

    class SeqNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(F, H, batch_first=True)
            self.head = nn.Sequential(
                nn.Linear(2 * H + pi_dim, 32), nn.ReLU(),
                nn.Dropout(0.2), nn.Linear(32, 3))

        def encode(self, seq):
            _, h = self.gru(seq)
            return h[-1]

        def forward(self, hs, as_, pi):
            f = [self.encode(hs), self.encode(as_)]
            if pi is not None:
                f.append(pi)
            return self.head(torch.cat(f, dim=1))

    return SeqNet()


def _train_eval_net(tr, va, data, *, use_pi, use_seq, H=24, epochs=12, bs=256):
    """Entrena la red en `tr` y predice `va`. use_seq/use_pi seleccionan inputs."""
    import torch
    import torch.nn as nn

    torch.manual_seed(SEED)
    F = data["F"]
    # estandariza pi y secuencias con stats de TRAIN (sin leak)
    pi = data["pi"]
    pi_sc = StandardScaler().fit(pi[tr])
    pi_t = pi_sc.transform(pi).astype(np.float32) if use_pi else None
    hs, as_ = data["home_seq"], data["away_seq"]
    flat = hs[tr].reshape(-1, F)
    mu, sd = flat.mean(0), flat.std(0) + 1e-6
    hs_n = ((hs - mu) / sd).astype(np.float32)
    as_n = ((as_ - mu) / sd).astype(np.float32)
    y = data["y"]

    if not use_seq:                       # solo pi -> logística (rápida y estable)
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(pi_sc.transform(pi[tr]), y[tr])
        p = clf.predict_proba(pi_sc.transform(pi[va]))
        out = np.zeros((len(va), 3))
        for j, c in enumerate(clf.classes_):
            out[:, c] = p[:, j]
        return out

    pi_dim = pi.shape[1] if use_pi else 0
    net = _make_net(F, H, pi_dim)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    Htr = torch.from_numpy(hs_n[tr]); Atr = torch.from_numpy(as_n[tr])
    Ytr = torch.from_numpy(y[tr])
    Ptr = torch.from_numpy(pi_t[tr]) if use_pi else None
    n = len(tr)
    net.train()
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            pib = Ptr[b] if use_pi else None
            logits = net(Htr[b], Atr[b], pib)
            loss = lossf(logits, Ytr[b])
            loss.backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        pib = torch.from_numpy(pi_t[va]) if use_pi else None
        logits = net(torch.from_numpy(hs_n[va]), torch.from_numpy(as_n[va]), pib)
        proba = torch.softmax(logits, dim=1).numpy()
    return proba


# ------------------------------- experimento --------------------------------
MODELS = {
    "pi-ratings solo": dict(use_pi=True, use_seq=False),
    "secuencia (GRU) sola": dict(use_pi=False, use_seq=True),
    "pi-ratings + secuencia": dict(use_pi=True, use_seq=True),
}


def run_for_N(N, n_splits, epochs) -> dict:
    data = build(N)
    y = data["y"]
    n = len(y)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    oof = {m: np.full((n, 3), np.nan) for m in MODELS}
    for tr, va in tscv.split(np.arange(n)):
        for m, cfg in MODELS.items():
            oof[m][va] = _train_eval_net(tr, va, data, epochs=epochs, **cfg)
    mask = ~np.isnan(oof["pi-ratings solo"][:, 0])
    res = {m: _metrics(y[mask], oof[m][mask]) for m in MODELS}
    return {"N": N, "n_oof": int(mask.sum()), "res": res}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-splits", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--Ns", type=int, nargs="+", default=[10, 20])
    args = ap.parse_args()

    try:
        import torch  # noqa: F401
    except ImportError:
        print("ERROR: falta torch. Instala con: pip install torch")
        sys.exit(1)

    blocks = []
    for N in args.Ns:
        print(f"\n=== N={N} (TimeSeriesSplit {args.n_splits} folds, "
              f"{args.epochs} epochs) ===")
        b = run_for_N(N, args.n_splits, args.epochs)
        for m, r in b["res"].items():
            print(f"  {m:>26}  RPS {r['rps']:.4f}  acc {r['acc']:.3f}")
        blocks.append(b)

    _write(blocks, args)
    print(f"\nReporte: {REPORT}")


def _write(blocks, args) -> None:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    md = ["# D. Forma como secuencia aprendida (GRU) vs pi-ratings\n",
          "Experimento AISLADO (no toca core/ ni Dixon-Coles): compara cómo "
          "representar la fuerza/forma de cada selección como input a un "
          "clasificador 1X2, con TimeSeriesSplit temporal y las métricas de core/ "
          "(RPS + accuracy). Menor RPS = mejor.\n",
          f"Protocolo: {args.n_splits} folds temporales, GRU oculto=24, "
          f"{args.epochs} epochs, Adam. Mismo conjunto OOF por N.\n"]
    overall_verdict = []
    for b in blocks:
        md.append(f"## N={b['N']} últimos partidos ({b['n_oof']} partidos OOF)\n")
        md.append("| Representación de fuerza | RPS | Accuracy |")
        md.append("|---|:-:|:-:|")
        res = b["res"]
        for m, r in res.items():
            md.append(f"| {m} | {r['rps']:.4f} | {r['acc']:.3f} |")
        pi = res["pi-ratings solo"]["rps"]
        seq = res["secuencia (GRU) sola"]["rps"]
        both = res["pi-ratings + secuencia"]["rps"]
        best = min(res, key=lambda k: res[k]["rps"])
        band = 0.0015            # banda de ruido en RPS: diferencias menores = empate
        md.append(f"\n- Mejor por RPS: **{best}** ({res[best]['rps']:.4f}). "
                  f"Diferencias < {band:.4f} RPS se consideran EMPATE (ruido).")
        if seq < pi - band:
            verdict = "la secuencia aprendida SOLA supera a pi-ratings"
        elif both < pi - band:
            verdict = ("la secuencia no gana sola, pero COMPLEMENTA "
                       "(pi+secuencia mejora a pi solo más allá del ruido)")
        else:
            verdict = ("la secuencia NO mejora a pi-ratings de forma apreciable "
                       "(diferencias dentro del ruido): pi-ratings se mantiene")
        md.append(f"- Lectura N={b['N']}: {verdict}.\n")
        overall_verdict.append(verdict)

    md.append("## Veredicto honesto\n")
    if all("NO mejora" in v for v in overall_verdict):
        md.append(
            "En ambos N, la **secuencia aprendida no le gana a los pi-ratings**. "
            "Coherente con la literatura de selecciones: con ~28k partidos pero "
            "muchas selecciones débiles y poca frecuencia de juego, un rating "
            "online recursivo (pi/Elo) ya captura casi toda la señal de forma; un "
            "GRU pequeño no encuentra estructura secuencial adicional que valga. "
            "Se reporta el resultado tal cual: **no se adopta** la secuencia en "
            "core/ (era una comparación, no un reemplazo).")
    else:
        md.append(
            "La secuencia aprendida aporta en algún N (ver arriba). Sigue siendo "
            "un experimento: **no** se cablea a core/ sin una validación temporal "
            "más amplia (más folds, tuning de N/hidden) y un chequeo de coste/"
            "beneficio frente a la simplicidad de pi-ratings.")
    md.append("\n> Limitación: GRU pequeño y pocos epochs por coste; el objetivo "
              "es la COMPARACIÓN relativa con pi-ratings bajo el mismo protocolo, "
              "no exprimir el mejor GRU posible. Las features de secuencia "
              "incluyen el Elo del rival/propio del momento, así que parte de la "
              "señal de 'fuerza' ya está disponible para el GRU.")
    REPORT.write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
