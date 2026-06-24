"""Sintetizador LLM acotado: decide un ajuste pequeño y validado en código.

Contrato (F del encargo):
  - El LLM devuelve SOLO JSON con: accion, magnitud, justificacion. NUNCA una
    probabilidad libre.
  - accion ∈ {sin_cambio, ajuste_leve, ajuste_fuerte, marcar_revision}.
  - magnitud ∈ [-0.15, 0.15] = ajuste a la probabilidad del resultado MÁS PROBABLE
    (se redistribuye en los otros dos). Límites DUROS en código: pase lo que pase
    el LLM, `clamp_decision` recorta al rango permitido por la acción.
  - justificacion: texto citando QUÉ señal del JSON motivó el ajuste.

Tope por acción (además del tope global ±0.15):
  - sin_cambio      -> magnitud forzada a 0.
  - ajuste_leve     -> |magnitud| ≤ 0.05.
  - ajuste_fuerte   -> |magnitud| ≤ 0.15.
  - marcar_revision -> magnitud forzada a 0 (señal para revisión humana, sin auto-ajuste).

Sin LLM / sin red / JSON inválido -> degrada a `sin_cambio` (no ajusta nada).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

ACTIONS = ("sin_cambio", "ajuste_leve", "ajuste_fuerte", "marcar_revision")
MAGNITUDE_HARD_CAP = 0.15            # tope global absoluto, inviolable
ACTION_CAP = {                       # tope por acción (|magnitud| máximo)
    "sin_cambio": 0.0, "ajuste_leve": 0.05,
    "ajuste_fuerte": 0.15, "marcar_revision": 0.0}
OUTCOMES = ("home", "draw", "away")
EPS = 1e-6


@dataclass
class SynthDecision:
    accion: str
    magnitud: float                  # YA acotada (post-clamp)
    justificacion: str
    raw_magnitud: Optional[float] = None   # lo que dijo el LLM, para auditar
    raw: Optional[dict] = None       # JSON crudo del LLM
    clamped: bool = False            # True si hubo que recortar

    def to_flat(self) -> dict:
        return {"synth_accion": self.accion, "synth_magnitud": self.magnitud,
                "synth_raw_magnitud": self.raw_magnitud,
                "synth_clamped": self.clamped,
                "synth_justificacion": (self.justificacion or "")[:500]}


# --------------------------- validación en código ---------------------------
def clamp_decision(accion, magnitud, justificacion, raw=None) -> SynthDecision:
    """Aplica los límites DUROS sin confiar en el LLM. Cualquier acción no válida
    cae a `marcar_revision` (conservador); la magnitud se recorta al tope global
    y luego al tope de la acción."""
    invalid_action = accion not in ACTIONS
    if invalid_action:
        accion = "marcar_revision"      # conservador ante acción desconocida
    raw_num = float(magnitud) if _isnum(magnitud) and np.isfinite(
        float(magnitud)) else None
    m = raw_num if raw_num is not None else 0.0
    # 1) tope global absoluto, 2) tope por acción
    m = float(np.clip(m, -MAGNITUDE_HARD_CAP, MAGNITUDE_HARD_CAP))
    m_capped = float(np.clip(m, -ACTION_CAP[accion], ACTION_CAP[accion]))
    clamped = invalid_action or (raw_num is None and magnitud is not None) or \
        (raw_num is not None and abs(raw_num - m_capped) > 1e-9)
    return SynthDecision(accion=accion, magnitud=m_capped,
                         justificacion=str(justificacion or ""),
                         raw_magnitud=raw_num, raw=raw, clamped=bool(clamped))


def _isnum(v) -> bool:
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def apply_adjustment(probs, decision: SynthDecision) -> np.ndarray:
    """Aplica el ajuste a la prob del resultado MÁS PROBABLE y redistribuye el
    delta proporcionalmente en los otros dos. Renormaliza y acota a (0,1).
    `marcar_revision`/`sin_cambio` (magnitud 0) -> devuelve probs sin cambio."""
    p = np.asarray(probs, float).copy()
    if decision.magnitud == 0.0:
        return p / p.sum()
    k = int(np.argmax(p))
    delta = float(decision.magnitud)
    new_k = float(np.clip(p[k] + delta, EPS, 1.0 - 2 * EPS))
    actual_delta = new_k - p[k]            # tras el clip, el delta efectivo
    others = [i for i in range(len(p)) if i != k]
    base = p[others].sum()
    if base <= EPS:                        # caso degenerado
        for i in others:
            p[i] = max(p[i] - actual_delta / len(others), EPS)
    else:
        for i in others:                   # quita proporcional a su peso
            p[i] = max(p[i] - actual_delta * (p[i] / base), EPS)
    p[k] = new_k
    return p / p.sum()


# ------------------------------- prompt / LLM -------------------------------
_SYS = (
    "Eres un AUDITOR de pronósticos de fútbol. Recibes dos modelos estadísticos "
    "(core y microsim) y un JSON de SEÑALES FACTUALES ya verificadas (lesionados, "
    "cambio de DT, dead rubber, fatiga, consenso). Tu trabajo NO es predecir el "
    "resultado ni dar probabilidades: es decidir si las señales justifican un "
    "ajuste PEQUEÑO a la probabilidad del resultado más probable del modelo. "
    "Respondes SOLO un objeto JSON con exactamente estas claves:\n"
    '  "accion": uno de ["sin_cambio","ajuste_leve","ajuste_fuerte","marcar_revision"]\n'
    '  "magnitud": número en [-0.15, 0.15] (positivo sube la prob del favorito, '
    "negativo la baja); 0 si sin_cambio o marcar_revision.\n"
    '  "justificacion": frase corta citando la señal EXACTA del JSON que motiva tu '
    "decisión (o por qué no ajustas).\n"
    "Sé conservador: si las señales son neutras o ausentes, usa sin_cambio y "
    "magnitud 0. Usa marcar_revision solo si las señales son contradictorias o "
    "alarmantes pero no sabes el signo del ajuste."
)


def build_prompt(core_probs, micro_probs, signals: Optional[dict]) -> str:
    fav = OUTCOMES[int(np.argmax(np.asarray(core_probs, float)))]
    payload = {
        "core_probs": {o: round(float(p), 4)
                       for o, p in zip(OUTCOMES, core_probs)},
        "microsim_probs": {o: round(float(p), 4)
                           for o, p in zip(OUTCOMES, micro_probs)},
        "resultado_mas_probable_core": fav,
        "senales_factuales": signals or {},
    }
    return ("Datos del partido (JSON de ENTRADA, NO lo repitas):\n"
            + json.dumps(payload, ensure_ascii=False, indent=2)
            + '\n\nDevuelve SOLO tu JSON de DECISIÓN con las claves '
            '{"accion","magnitud","justificacion"}. No copies el JSON de entrada.')


# ----------------------- proveedor LM Studio propio -------------------------
def _load_dotenv(path: Path = ROOT / ".env") -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def _default_provider(*, timeout_s: float = 120.0
                      ) -> Optional[Callable[[str], Optional[str]]]:
    """Provider LM Studio con el system prompt del SINTETIZADOR (`_SYS`).

    No se reutiliza `llm_features.lmstudio_provider` porque ese fija su propio
    system prompt de EXTRACTOR (el modelo acababa repitiendo el JSON de entrada).
    Aquí el system prompt es el de AUDITOR/decisión."""
    _load_dotenv()
    url = os.environ.get("LMSTUDIO_URL", "http://localhost:1234/v1")
    if not url:
        return None
    api_key = os.environ.get("LMSTUDIO_API_KEY") or None
    model = os.environ.get("LMSTUDIO_MODEL")

    def _headers() -> dict:
        h = {"Content-Type": "application/json"}
        if api_key:
            h["Authorization"] = f"Bearer {api_key}"
        return h

    def _provider(prompt: str) -> Optional[str]:
        try:
            import requests
        except ImportError:
            return None
        m = model
        try:                                   # autodetecta el modelo cargado
            r = requests.get(f"{url}/models", headers=_headers(), timeout=10)
            ids = [d["id"] for d in r.json().get("data", [])]
            chat = [i for i in ids if "embed" not in i.lower()]
            if m not in ids and chat:
                m = chat[0]
        except Exception:
            pass
        base = {"model": m, "temperature": 0, "max_tokens": 400,
                "messages": [{"role": "system", "content": _SYS},
                             {"role": "user", "content": prompt}]}
        for payload in ({**base, "response_format": {"type": "json_object"}}, base):
            try:
                r = requests.post(f"{url}/chat/completions", headers=_headers(),
                                  json=payload, timeout=timeout_s)
                if r.status_code != 200:
                    continue
                return r.json()["choices"][0]["message"]["content"]
            except Exception:
                continue
        return None

    return _provider


def _parse_json(text: str) -> Optional[dict]:
    if not text:
        return None
    t = text.strip()
    t = re.sub(r"^```(?:json)?|```$", "", t, flags=re.MULTILINE).strip()
    m = re.search(r"\{.*\}", t, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def synthesize(core_probs, micro_probs, signals: Optional[dict] = None, *,
               provider: Optional[Callable[[str], Optional[str]]] = None
               ) -> SynthDecision:
    """Pide la decisión al LLM (LM Studio por defecto) y la ACOTA en código.

    `signals`: dict del schema de llm_features YA validado por roster.py.
    `provider`: callable(prompt)->str|None; si None, usa LM Studio local. Sin
    proveedor / sin red / JSON inválido -> decisión `sin_cambio` (degradación).
    """
    if provider is None:
        provider = _default_provider()
    if provider is None:
        return SynthDecision("sin_cambio", 0.0,
                             "LLM no disponible: sin ajuste.", clamped=False)

    prompt = build_prompt(core_probs, micro_probs, signals)
    try:
        text = provider(prompt)
    except Exception:
        text = None
    data = _parse_json(text or "")
    if data is None:
        return SynthDecision("sin_cambio", 0.0,
                             "Respuesta del LLM no parseable: sin ajuste.",
                             clamped=False)
    return clamp_decision(data.get("accion"), data.get("magnitud"),
                          data.get("justificacion"), raw=data)
