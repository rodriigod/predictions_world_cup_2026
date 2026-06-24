# experiments/club_stress_test/ — ¿el stacking funciona con MUCHOS datos?

**Diagnóstico de metodología, NO producción del Mundial.** No toca `core/` ni
`microsim/`: es un pipeline paralelo y autocontenido.

Pregunta: con solo 384 partidos de Mundial no sabemos si el stacking falla por
*concepto* (modelos correlacionados) o por *tamaño de muestra*. Aquí se corre el
MISMO esquema (dos modelos base de fuerza + meta-modelo logístico) sobre una
liga de clubes con **decenas de miles de partidos** (football match data,
top-5 ligas europeas 2000-2025, `files/cache/club_matches.csv`).

- **Modelo A (análogo a core)**: logística 1X2 sobre la diferencia de ELO
  pre-partido.
- **Modelo B (análogo a microsim)**: fuerza por GOLES — medias móviles
  leak-free de goles a favor/en contra (ataque/defensa) + forma reciente. Es
  una señal DISTINTA del ELO (el microsim del Mundial no pudo serlo: no hay
  valores de mercado históricos).
- **Stacking**: meta-modelo logístico sobre las 1X2 de A y B, evaluado OOF con
  TimeSeriesSplit.

Correr: `python experiments/club_stress_test/run_stress_test.py`
→ `results/reports/club_stress_test.md`.
