# Comparación SIN vs CON datos 2026 — índice maestro

Dos regímenes, los mismos 3 modelos en cada uno:

- **SIN datos** = pipeline de producción con ratings **pre-torneo** (core, microsim, ensemble; el LLM entra dentro del ensemble).
- **CON datos** = los mismos modelos pero con los ratings **online-updated** con la 1ª ronda real (módulo `online_learning/`).

## 1. Paneles por partido (matriz de marcadores + 1X2 + top-10)

72 partidos de grupos × 6 modelos = **432 imágenes**.
- Índice: [`results/match_panels/INDEX.md`](../match_panels/INDEX.md)
- Por partido: `results/match_panels/<Local_vs_Visita>/<modelo>_<sin|con>_datos.png`
- Cada panel: matriz 0-5×0-5 (casilla resaltada = marcador más probable), barras 1X2, top-10 marcadores.

## 2a. Predicciones de los PRÓXIMOS partidos (48 sin jugar)

[`proximos_partidos_predicciones.md`](proximos_partidos_predicciones.md) — tabla consolidada: cada partido sin jugar × 6 modelos (P(1)/P(X)/P(2) + marcador). Las imágenes de cada uno están en `results/match_panels/`.

## 2b. Tabla comparativa en los partidos jugados (24)

[`model_comparison_online.md`](model_comparison_online.md) — P(1X2) + marcador + puntos 5/3/0 por modelo, con totales:

| Modelo | Puntos | Aciertos | RPS |
|---|:-:|:-:|:-:|
| core · sin | 40 | 12/24 | 0.1975 |
| core · con | 43 | 13/24 | 0.1697 |
| microsim · sin | 37 | 11/24 | 0.2296 |
| microsim · con | 33 | 11/24 | 0.2071 |
| ensemble · sin | 38 | 12/24 | 0.1960 |
| ensemble · con | 41 | 13/24 | 0.1718 |

> ⚠️ "con" en la 1ª ronda es **in-sample** (los ratings ya incluyen ese resultado): sirve para ver el ajuste, no como predicción honesta. La comparación limpia será en las rondas que faltan.

## 3. Simulación del torneo completo (hasta la final)

[`tournament_comparison.md`](tournament_comparison.md) — 1500 torneos Monte Carlo por modelo, cuadro oficial 2026. Tablas por ronda (16avos→campeón) en [`tournament_sim/`](tournament_sim/).

**El efecto del online learning, visible en las probabilidades de campeón:**

| | core·sin → core·con | ensemble·sin → ensemble·con |
|---|---|---|
| **España** (0-0 vs Cabo Verde) | 20.5% → **15.0%** ↓ | 17.1% → **14.3%** ↓ |
| **Argentina** (3-0 vs Argelia) | 16.0% → **20.2%** ↑ | 17.2% → **18.7%** ↑ |

La decepción de España y la goleada de Argentina en la 1ª ronda **reordenan los favoritos** — justo lo que el módulo `online_learning` está diseñado para capturar.

## Cómo regenerar

```bash
python scripts/generate_match_panels.py        # 432 paneles
python scripts/model_comparison_online.py       # tabla sin/con (jugados)
python scripts/tournament_compare.py            # simulación 6 modelos
```
