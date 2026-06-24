# online_learning — ratings "2026-ajustados" (en paralelo a producción)

Actualiza la fuerza de cada selección con los **resultados reales del Mundial
2026** a medida que se juegan, **sin tocar** `core/`, `microsim/`,
`llm_features/` ni `ensemble/`. Produce ratings alternativos usables como
reemplazo *drop-in* de los pre-torneo para los partidos que faltan.

> **Autocontenido**: ningún archivo de producción importa este paquete. Borrar
> `online_learning/` deja el resto del repo intacto (verificado).

## Secciones

| | Qué hace | API |
|---|---|---|
| **A** | ELO online (arranca del ELO pre-torneo de core/, K=30 grupos / 40 eliminatorias) | `get_elo_2026(team)` |
| **B** | pi-ratings online (ataque/defensa, mismo update que core/) | `get_pi_2026(team) -> (att, dfn)` |
| **C** | Fuerza Bayesiana (Gamma-Poisson; media + IC80%) | `get_strength_posterior(team)` |
| **D** | Detección de sorpresas + sesgo de favoritos | `build_surprise_log()`, `detect_favorite_bias()` |
| **E** | Ensemble con ratings 2026 | `predict_final_updated(h, a, date)` |
| **F** | Mantenimiento (agregar partido, recomputar todo) | `update_results.py` |

## Datos

`data/results_2026.csv` — formato `date,home_team,away_team,home_goals,away_goals,stage`
(stage ∈ group/r32/r16/qf/sf/final). Nombres en español o inglés (se normalizan).

> **Nota honesta sobre la data**: el prompt hablaba de "24 partidos de la primera
> ronda", pero solo **2** están como resultado REAL en el dataset del repo
> (martj42): `México 2-0 Sudáfrica` y `Corea del Sur 2-1 Rep. Checa` (2026-06-11).
> El CSV se siembra con esos 2 reales. **No se inventan los otros 22** — se
> agregan con `update_results.py --add` a medida que se jueguen (o pega aquí los
> 24 marcadores y recargo).

## Uso

```bash
# agregar un partido y recomputar ELO+pi+bayes+sorpresas+reporte:
python online_learning/update_results.py --add 2026-06-12 España "Cabo Verde" 1 2 group

# recomputar reportes (y anotar prob_core_updated en los live logs):
python online_learning/update_results.py --rebuild --annotate-logs

# ver estado actual de los equipos con datos:
python online_learning/update_results.py --state
```

```python
import online_learning as ol
ol.get_elo_2026("España")                 # ELO drop-in actualizado
ol.get_strength_posterior("Brasil")["off"]  # {mean, lo, hi, n}
ol.predict_final_updated("México", "Sudáfrica", "2026-06-20")
```

## Salidas

- `data/surprise_log.csv` — fecha, equipos, resultado, prob_core_pre, surprise_score.
- `results/reports/online_learning_round.md` — ELO antes/después, top movimientos,
  accuracy/RPS original vs actualizado.
- columna `prob_core_updated` en `results/live_log/*.csv` (con `--annotate-logs`).

## Decisiones documentadas

- **Con 1 partido por equipo, los ratings se mueven poco** — correcto, no es señal
  suficiente. El Bayesiano (κ=5 partidos de prior) se mueve **menos** que el ELO,
  por diseño.
- **K-factor**: core/ usa K=60 para Mundial (con multiplicador de goleada); aquí
  se usa el 30/40 conservador del encargo para un actualizador de un solo torneo.
  Configurable en `elo_online.compute_elo`.
- **Sede neutral** por defecto (ventaja local 0); los anfitriones podrían llevar
  un bonus si se quisiera.
