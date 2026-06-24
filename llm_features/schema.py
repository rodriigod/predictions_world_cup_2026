"""Schema de salida ESTRUCTURADO de las features de LLM por partido.

IMPORTANTE: estas son SEÑALES OBJETIVAS Y VERIFICABLES extraídas de la web,
NO una predicción del resultado. El LLM nunca dice quién gana; solo rellena
estos campos con hechos buscables. Cualquier campo sin información confiable
queda en `None` (no se inventa).

Cada feature relevante se reporta por equipo (`home`/`away`) cuando aplica.
El resultado final que consume el `ensemble/` es un dict plano y serializable
(ver `MatchFeatures.to_flat_dict`), apto para añadirse como columnas extra.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass
class TeamSignals:
    """Señales por equipo. Todos los campos son opcionales (None = sin dato)."""
    # Jugadores titulares/clave confirmados FUERA por lesión (lista de nombres).
    lesionados_clave: Optional[list[str]] = None
    # ¿Hubo cambio de entrenador en los últimos ~3 meses?
    cambio_dt_reciente: Optional[bool] = None
    # % de analistas/medios reconocidos que FAVORECEN a este equipo (0-100),
    # agregando predicciones publicadas reales (no generadas por el LLM).
    consenso_expertos_pct: Optional[float] = None
    # Fatiga de viaje desde el último partido jugado.
    fatiga_viaje_km: Optional[float] = None
    fatiga_husos_horarios: Optional[int] = None


@dataclass
class MatchFeatures:
    """Features estructuradas de UN partido. `home`/`away` siguen al fixture."""
    # --- identidad (obligatoria) ---
    team_home: str
    team_away: str
    match_date: str                       # ISO 'YYYY-MM-DD'

    # --- señales por equipo ---
    home: TeamSignals = field(default_factory=TeamSignals)
    away: TeamSignals = field(default_factory=TeamSignals)

    # --- señal a nivel partido ---
    # ¿Es un partido intrascendente (fase de grupos) porque alguno de los dos
    # ya está clasificado o eliminado matemáticamente ANTES de jugarlo?
    dead_rubber: Optional[bool] = None

    # --- procedencia ---
    source: str = "unavailable"           # "llm_web_search" | "cache" | "unavailable"
    retrieved_at: Optional[str] = None     # ISO timestamp de la extracción
    provider: Optional[str] = None         # p.ej. "gemini:gemini-flash-latest"
    # Notas/citas devueltas por el LLM (trazabilidad); no entran al ensemble.
    notes: Optional[str] = None

    # ------------------------------------------------------------------ utils
    @classmethod
    def null(cls, team_home: str, team_away: str, match_date: str,
             source: str = "unavailable") -> "MatchFeatures":
        """Esqueleto con todas las señales en None (cuando no hay extracción)."""
        return cls(team_home=team_home, team_away=team_away,
                   match_date=str(match_date), source=source)

    def to_dict(self) -> dict:
        """Dict anidado, serializable a JSON tal cual."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "MatchFeatures":
        home = TeamSignals(**(d.get("home") or {}))
        away = TeamSignals(**(d.get("away") or {}))
        known = {k for k in cls.__dataclass_fields__ if k not in ("home", "away")}
        rest = {k: v for k, v in d.items() if k in known}
        return cls(home=home, away=away, **rest)

    def to_flat_dict(self) -> dict:
        """Vista PLANA (una fila), lista para ser columnas extra del ensemble.
        Prefija las señales por equipo con home_/away_."""
        flat = {"team_home": self.team_home, "team_away": self.team_away,
                "match_date": self.match_date, "dead_rubber": self.dead_rubber,
                "features_source": self.source}
        for side, sig in (("home", self.home), ("away", self.away)):
            d = asdict(sig)
            for k, v in d.items():
                # las listas (lesionados) se resumen a un conteo para columna escalar
                if isinstance(v, list):
                    flat[f"{side}_{k}_n"] = len(v)
                else:
                    flat[f"{side}_{k}"] = v
        return flat
